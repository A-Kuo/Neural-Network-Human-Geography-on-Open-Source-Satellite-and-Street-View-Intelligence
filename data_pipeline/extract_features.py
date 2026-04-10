"""
extract_features.py — ResNet-152 feature extraction from Street View images.

Ethics rules enforced (NON-NEGOTIABLE):
1. Original images DELETED after feature extraction (del + os.remove)
2. Output stores only: tract_id, image_index, feature_vector, extraction_date
3. No lat/lon, no address, no filename stored in feature table
4. Feature vectors are tract-aggregated before any income data joins

Pipeline:
    data/raw/streetview/<tract_id>/*.jpg
    → ResNet-152 embeddings (2048-dim)
    → DELETE original images
    → data/processed/image_features/<tract_id>_features.npy
    → data/processed/tract_image_features.parquet (aggregated by tract)

Usage:
    python extract_features.py \
        --image-dir data/raw/streetview/ \
        --output-dir data/processed/image_features/ \
        [--batch-size 32] [--device cuda]
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Constants ───────────────────────────────────────────────────────────────────

RESNET_FEATURE_DIM = 2048  # ResNet-152 penultimate layer output
MIN_IMAGES_PER_TRACT = 10  # Minimum for privacy/reliability (see ethics rules)
IMAGE_SIZE = (224, 224)  # ResNet input size

# ImageNet normalization (ResNet was trained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Image preprocessing ─────────────────────────────────────────────────────────

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class TractImageDataset(Dataset):
    """
    Dataset for images in a single Census tract directory.

    Loads images by path but does NOT store lat/lon or address metadata.
    Only tract_id and sequential index are tracked.
    """

    def __init__(self, image_paths: list[Path], transform=IMAGE_TRANSFORM):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, idx
        except Exception as e:
            logger.warning(f"Failed to load image {path.name}: {e} — returning zeros")
            return torch.zeros(3, *IMAGE_SIZE), idx


# ── Model setup ─────────────────────────────────────────────────────────────────


def load_resnet152(device: str = "cpu") -> nn.Module:
    """
    Load pretrained ResNet-152. Remove final classification layer to get embeddings.

    Output: 2048-dimensional feature vectors (average pooled spatial features).
    Using pretrained weights removes need for expensive fine-tuning at this stage.
    """
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    # Remove the classification head — keep everything up to avgpool
    model.fc = nn.Identity()
    model.eval()
    model = model.to(device)
    logger.info(f"ResNet-152 loaded on {device} ({RESNET_FEATURE_DIM}-dim embeddings)")
    return model


# ── Feature extraction ──────────────────────────────────────────────────────────


@torch.no_grad()
def extract_tract_features(
    image_paths: list[Path],
    model: nn.Module,
    device: str,
    batch_size: int = 32,
) -> Optional[np.ndarray]:
    """
    Extract ResNet-152 embeddings for all images in one tract.

    Returns array of shape (n_images, 2048) or None if insufficient images.
    """
    if len(image_paths) < MIN_IMAGES_PER_TRACT:
        return None

    dataset = TractImageDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    for batch_tensors, _ in loader:
        batch_tensors = batch_tensors.to(device)
        features = model(batch_tensors)  # (batch, 2048)
        all_features.append(features.cpu().numpy())
        del batch_tensors, features
        gc.collect()

    return np.vstack(all_features)


def _delete_tract_images(image_paths: list[Path], dry_run: bool = False) -> int:
    """
    Delete original Street View images after feature extraction.

    This is MANDATORY per ethics rules — we never retain identifiable images.
    Returns count of deleted files.
    """
    deleted = 0
    for path in image_paths:
        if path.exists():
            if not dry_run:
                os.remove(path)
            deleted += 1
    if deleted:
        action = "Would delete" if dry_run else "Deleted"
        logger.debug(f"  {action} {deleted} original images (ethics: no image retention).")
    return deleted


# ── Aggregation ─────────────────────────────────────────────────────────────────


def aggregate_tract_features(features: np.ndarray) -> dict:
    """
    Aggregate per-image features to tract-level statistics.

    Privacy rule: we store summary statistics, not individual image vectors.
    (Individual vectors are saved separately per-tract for model training,
    but the joined dataset only uses aggregated stats.)

    Returns dict with mean and std of each feature dimension.
    """
    return {
        "mean": features.mean(axis=0),  # (2048,)
        "std": features.std(axis=0),  # (2048,)
        "n_images": len(features),
    }


# ── Main extraction loop ────────────────────────────────────────────────────────


def extract_all(
    image_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
    delete_after_extraction: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Extract ResNet features for all Census tracts in image_dir.

    Directory structure expected:
        image_dir/
            <tract_id>/
                <tract_id>_0001_h0.jpg
                <tract_id>_0001_h90.jpg
                ...

    Args:
        image_dir: Root directory of fetched images
        output_dir: Where to save per-tract .npy feature arrays
        device: "cpu" or "cuda"
        batch_size: Images per forward pass
        delete_after_extraction: Delete JPEGs after embedding (REQUIRED for ethics)
        dry_run: If True, simulate deletion without deleting

    Returns:
        DataFrame with tract-level aggregate feature stats (for audit/logging)
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not delete_after_extraction:
        logger.warning(
            "delete_after_extraction=False — original images will be RETAINED. "
            "This violates ethics rules unless you have a documented exception."
        )

    model = load_resnet152(device=device)

    tract_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir()])
    logger.info(f"Processing {len(tract_dirs)} tract directories...")

    summary_records = []
    extraction_date = datetime.now().isoformat()

    for tract_dir in tract_dirs:
        tract_id = tract_dir.name
        output_npy = output_dir / f"{tract_id}_features.npy"
        meta_path = output_dir / f"{tract_id}_meta.json"

        image_paths = sorted(tract_dir.glob("*.jpg"))

        if output_npy.exists():
            logger.debug(f"Tract {tract_id}: already extracted — skipping.")
            if delete_after_extraction and image_paths:
                _delete_tract_images(image_paths, dry_run=dry_run)
            continue

        if len(image_paths) < MIN_IMAGES_PER_TRACT:
            logger.info(
                f"Tract {tract_id}: only {len(image_paths)} images "
                f"(< {MIN_IMAGES_PER_TRACT} minimum) — skipping (sparse)."
            )
            summary_records.append(
                {
                    "tract_id": tract_id,
                    "n_images": len(image_paths),
                    "status": "sparse_skipped",
                    "extraction_date": extraction_date,
                }
            )
            continue

        logger.info(f"Tract {tract_id}: extracting features from {len(image_paths)} images...")
        try:
            features = extract_tract_features(image_paths, model, device, batch_size)
        except Exception as e:
            logger.error(f"Tract {tract_id}: extraction failed — {e}")
            summary_records.append(
                {
                    "tract_id": tract_id,
                    "n_images": len(image_paths),
                    "status": f"error: {e}",
                    "extraction_date": extraction_date,
                }
            )
            continue

        if features is None:
            continue

        # Save per-tract feature matrix (used during model training)
        # Shape: (n_images, 2048) — no lat/lon, just tract_id in filename
        np.save(output_npy, features)

        # Save lightweight metadata (no PII)
        agg = aggregate_tract_features(features)
        meta = {
            "tract_id": tract_id,
            "n_images": int(agg["n_images"]),
            "feature_dim": RESNET_FEATURE_DIM,
            "extraction_date": extraction_date,
            "model": "resnet152_imagenet",
            # Store mean/std as lists for JSON serialization
            "mean_feature_norm": float(np.linalg.norm(agg["mean"])),
            "std_feature_norm": float(np.linalg.norm(agg["std"])),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        summary_records.append(
            {
                "tract_id": tract_id,
                "n_images": int(agg["n_images"]),
                "status": "ok",
                "extraction_date": extraction_date,
            }
        )

        # ETHICS RULE: Delete original images NOW
        if delete_after_extraction:
            n_deleted = _delete_tract_images(image_paths, dry_run=dry_run)
            logger.debug(f"  Ethics: deleted {n_deleted} source images for tract {tract_id}.")

        logger.info(
            f"  → {tract_id}: {agg['n_images']} images, " f"features saved to {output_npy.name}"
        )

    # Build aggregate features DataFrame (tract-level means for joining)
    logger.info("Building tract-level aggregate feature table...")
    agg_rows = []
    for npy_file in sorted(output_dir.glob("*_features.npy")):
        tract_id = npy_file.stem.replace("_features", "")
        features = np.load(npy_file)
        mean_vec = features.mean(axis=0)
        # Column names: img_feat_0, img_feat_1, ..., img_feat_2047
        row = {"tract_id": tract_id, "n_images": len(features)}
        for i, v in enumerate(mean_vec):
            row[f"img_feat_{i}"] = float(v)
        agg_rows.append(row)

    if agg_rows:
        agg_df = pd.DataFrame(agg_rows)
        agg_path = Path("data/processed") / "tract_image_features.parquet"
        Path("data/processed").mkdir(exist_ok=True)
        agg_df.to_parquet(agg_path, index=False)
        logger.info(f"Tract-level feature table: {len(agg_df)} tracts → {agg_path}")

    # Save extraction log
    summary_df = pd.DataFrame(summary_records)
    audit_path = Path("data/audit") / "feature_extraction_log.csv"
    Path("data/audit").mkdir(exist_ok=True)
    summary_df.to_csv(audit_path, index=False)

    n_ok = (summary_df["status"] == "ok").sum() if not summary_df.empty else 0
    logger.info(f"\nExtraction complete: {n_ok}/{len(tract_dirs)} tracts processed.")
    return summary_df


# ── CLI ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract ResNet-152 features from Street View images"
    )
    parser.add_argument(
        "--image-dir",
        default="data/raw/streetview/",
        help="Root dir of fetched images (tract subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/image_features/",
        help="Output dir for .npy feature arrays",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Compute device"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="DANGER: Retain original images (ethics violation unless justified)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate image deletion without actually deleting"
    )
    args = parser.parse_args()

    if args.no_delete:
        logger.warning(
            "--no-delete set: images WILL be retained. "
            "Document your ethics justification in ETHICS.md."
        )

    summary = extract_all(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        delete_after_extraction=not args.no_delete,
        dry_run=args.dry_run,
    )
    ok = summary[summary["status"] == "ok"].shape[0] if not summary.empty else 0
    print(f"\nDone. {ok} tracts extracted. Log → data/audit/feature_extraction_log.csv")
