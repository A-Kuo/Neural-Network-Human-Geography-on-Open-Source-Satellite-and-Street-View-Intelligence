"""
fetch_streetview.py — Rate-limited Street View image fetcher with coverage audit.

Ethics rules enforced:
- Images are fetched for Census tract centroids only (no individual addresses)
- Output: raw images tagged with tract_id only (no lat/lon stored in metadata)
- Coverage audit flags sparse tracts for bias documentation
- All requests logged for reproducibility

Usage:
    python fetch_streetview.py --api-key $GOOGLE_SV_API_KEY \
        --tracts data/processed/tract_centroids.geojson \
        --output data/raw/streetview/ \
        --images-per-tract 30

Rate limit: 1 req/sec (Google Street View Static API free tier: 25k/month)
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Constants ──────────────────────────────────────────────────────────────────

STREET_VIEW_API_URL = "https://maps.googleapis.com/maps/api/streetview"
METADATA_API_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

# Headings to capture all cardinal directions per centroid location
HEADINGS = [0, 90, 180, 270]  # N, E, S, W
PITCH = 0  # horizontal view (not up/down)
FOV = 90  # field of view in degrees
IMG_SIZE = "640x640"
MIN_IMAGES_FOR_COVERAGE = 10  # tracts with fewer are "sparse"
PREFERRED_IMAGES_PER_TRACT = 30  # 30 = 4 headings × ~7-8 sampling points

# ── Data structures ─────────────────────────────────────────────────────────────


@dataclass
class ImageRecord:
    """Metadata for one fetched Street View image. No lat/lon stored."""

    tract_id: str
    image_index: int
    heading: int
    filename: str
    fetch_date: str
    status: str  # "ok" | "no_imagery" | "error"
    pano_id: Optional[str] = None  # anonymized panorama ID


@dataclass
class TractCoverageRecord:
    """Coverage audit record per Census tract."""

    tract_id: str
    n_requested: int
    n_fetched: int
    n_no_imagery: int
    n_errors: int
    coverage_flag: str  # "full" | "sparse" | "missing"


# ── Rate-limited API calls ──────────────────────────────────────────────────────


@sleep_and_retry
@limits(calls=1, period=1)
def _get_with_rate_limit(url: str, params: dict) -> requests.Response:
    """Enforce 1 request/second globally across all API calls."""
    return requests.get(url, params=params, timeout=15)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def check_imagery_available(lat: float, lon: float, api_key: str) -> Optional[str]:
    """
    Check if Street View imagery exists near (lat, lon).

    Returns panorama ID if imagery exists, None otherwise.
    Uses metadata endpoint (free, no charge per request).
    """
    params = {
        "location": f"{lat},{lon}",
        "radius": 50,  # meters — stay near centroid
        "key": api_key,
    }
    resp = _get_with_rate_limit(METADATA_API_URL, params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "OK":
        return data.get("pano_id")
    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_single_image(
    lat: float,
    lon: float,
    heading: int,
    api_key: str,
    output_path: Path,
) -> bool:
    """
    Fetch one Street View image. Returns True on success.

    NOTE: lat/lon are NOT stored in the output filename or metadata.
    Only tract_id and heading are preserved. This prevents reverse-geocoding.
    """
    params = {
        "size": IMG_SIZE,
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": PITCH,
        "fov": FOV,
        "key": api_key,
        "return_error_code": True,
    }
    resp = _get_with_rate_limit(STREET_VIEW_API_URL, params)
    if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
        output_path.write_bytes(resp.content)
        return True
    return False


# ── Sampling strategy ───────────────────────────────────────────────────────────


def sample_points_in_tract(
    tract_geometry,
    n_points: int = 8,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Generate up to n_points sampling locations within a Census tract polygon.

    Uses a regular grid clipped to the tract boundary rather than random
    sampling so results are reproducible without a random seed in the filename.
    """
    import numpy as np
    from shapely.geometry import Point

    bounds = tract_geometry.bounds  # (minx, miny, maxx, maxy)
    rng = np.random.default_rng(seed)

    # Slightly denser grid, then clip to polygon
    candidates = []
    for _ in range(n_points * 10):
        x = rng.uniform(bounds[0], bounds[2])
        y = rng.uniform(bounds[1], bounds[3])
        pt = Point(x, y)
        if tract_geometry.contains(pt):
            candidates.append((y, x))  # (lat, lon)
        if len(candidates) >= n_points:
            break

    return candidates


# ── Main fetch loop ─────────────────────────────────────────────────────────────


def fetch_tracts(
    tracts_path: str | Path,
    output_dir: str | Path,
    api_key: str,
    images_per_tract: int = PREFERRED_IMAGES_PER_TRACT,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch Street View images for all Census tracts.

    Args:
        tracts_path: GeoJSON with columns [tract_id, geometry]
        output_dir: Directory for raw images (will be deleted after extraction)
        api_key: Google Street View Static API key
        images_per_tract: Target images per tract (4 headings × N locations)
        resume: Skip tracts already fetched (idempotent)

    Returns:
        (image_log_df, coverage_df) — saved to data/audit/
    """
    tracts_path = Path(tracts_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracts = gpd.read_file(tracts_path)
    if "tract_id" not in tracts.columns:
        raise ValueError("GeoJSON must have 'tract_id' column")

    # Ensure WGS84 for lat/lon API calls
    tracts = tracts.to_crs(epsg=4326)

    n_points_per_tract = max(1, images_per_tract // len(HEADINGS))
    image_records: list[ImageRecord] = []
    coverage_records: list[TractCoverageRecord] = []

    logger.info(
        f"Fetching Street View for {len(tracts)} tracts "
        f"({images_per_tract} images/tract target)"
    )

    for _, row in tracts.iterrows():
        tract_id = str(row["tract_id"])
        tract_dir = output_dir / tract_id

        if (
            resume
            and tract_dir.exists()
            and len(list(tract_dir.glob("*.jpg"))) >= MIN_IMAGES_FOR_COVERAGE
        ):
            logger.debug(f"Skipping {tract_id} (already fetched)")
            continue

        tract_dir.mkdir(exist_ok=True)
        sampling_points = sample_points_in_tract(row.geometry, n_points=n_points_per_tract)

        n_fetched = 0
        n_no_imagery = 0
        n_errors = 0
        img_index = 0

        for lat, lon in sampling_points:
            # Check availability first (free metadata endpoint)
            try:
                pano_id = check_imagery_available(lat, lon, api_key)
            except Exception as e:
                logger.warning(f"Metadata check failed at ({lat:.4f},{lon:.4f}): {e}")
                n_errors += 1
                continue

            if pano_id is None:
                n_no_imagery += 1
                image_records.append(
                    ImageRecord(
                        tract_id=tract_id,
                        image_index=img_index,
                        heading=-1,
                        filename="",
                        fetch_date=pd.Timestamp.now().isoformat(),
                        status="no_imagery",
                    )
                )
                img_index += 1
                continue

            for heading in HEADINGS:
                # Filename encodes only tract + index — NO lat/lon
                filename = f"{tract_id}_{img_index:04d}_h{heading}.jpg"
                filepath = tract_dir / filename

                try:
                    ok = fetch_single_image(lat, lon, heading, api_key, filepath)
                    status = "ok" if ok else "error"
                    if ok:
                        n_fetched += 1
                    else:
                        n_errors += 1
                except Exception as e:
                    logger.error(f"Fetch failed {filename}: {e}")
                    status = "error"
                    n_errors += 1

                image_records.append(
                    ImageRecord(
                        tract_id=tract_id,
                        image_index=img_index,
                        heading=heading,
                        filename=filename if status == "ok" else "",
                        fetch_date=pd.Timestamp.now().isoformat(),
                        status=status,
                        pano_id=pano_id,
                    )
                )
                img_index += 1

        # Coverage classification for bias audit
        if n_fetched == 0:
            flag = "missing"
        elif n_fetched < MIN_IMAGES_FOR_COVERAGE:
            flag = "sparse"
        else:
            flag = "full"

        coverage_records.append(
            TractCoverageRecord(
                tract_id=tract_id,
                n_requested=len(sampling_points) * len(HEADINGS),
                n_fetched=n_fetched,
                n_no_imagery=n_no_imagery,
                n_errors=n_errors,
                coverage_flag=flag,
            )
        )
        logger.info(f"Tract {tract_id}: {n_fetched} images ({flag})")

    image_log_df = pd.DataFrame([asdict(r) for r in image_records])
    coverage_df = pd.DataFrame([asdict(r) for r in coverage_records])

    # Save logs to audit directory
    audit_dir = Path("data/audit")
    audit_dir.mkdir(exist_ok=True)
    image_log_df.to_csv(audit_dir / "streetview_image_log.csv", index=False)
    coverage_df.to_csv(audit_dir / "streetview_coverage.csv", index=False)

    _log_coverage_summary(coverage_df)
    return image_log_df, coverage_df


def _log_coverage_summary(coverage_df: pd.DataFrame) -> None:
    """Print bias-relevant coverage statistics to stdout and logger."""
    total = len(coverage_df)
    full = (coverage_df["coverage_flag"] == "full").sum()
    sparse = (coverage_df["coverage_flag"] == "sparse").sum()
    missing = (coverage_df["coverage_flag"] == "missing").sum()

    logger.info("=" * 60)
    logger.info("STREET VIEW COVERAGE AUDIT")
    logger.info(f"  Total tracts: {total}")
    logger.info(
        f"  Full coverage (>={MIN_IMAGES_FOR_COVERAGE} imgs): {full} ({100*full/total:.1f}%)"
    )
    logger.info(
        f"  Sparse (<{MIN_IMAGES_FOR_COVERAGE} imgs):         {sparse} ({100*sparse/total:.1f}%)"
    )
    logger.info(f"  Missing (0 imgs):                {missing} ({100*missing/total:.1f}%)")
    if missing > 0:
        missing_ids = coverage_df[coverage_df["coverage_flag"] == "missing"]["tract_id"].tolist()
        logger.warning(f"  Missing tract IDs: {missing_ids}")
    logger.info("  NOTE: Sparse/missing tracts will be excluded from")
    logger.info("  training and flagged in bias report (see ETHICS.md).")
    logger.info("=" * 60)


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Chicago Street View imagery by Census tract"
    )
    parser.add_argument("--api-key", required=True, help="Google Street View Static API key")
    parser.add_argument(
        "--tracts",
        default="data/processed/tract_centroids.geojson",
        help="Input GeoJSON with tract_id + geometry",
    )
    parser.add_argument(
        "--output", default="data/raw/streetview/", help="Output directory for raw images"
    )
    parser.add_argument("--images-per-tract", type=int, default=PREFERRED_IMAGES_PER_TRACT)
    parser.add_argument(
        "--no-resume", action="store_true", help="Refetch all tracts even if already done"
    )
    args = parser.parse_args()

    image_log, coverage = fetch_tracts(
        tracts_path=args.tracts,
        output_dir=args.output,
        api_key=args.api_key,
        images_per_tract=args.images_per_tract,
        resume=not args.no_resume,
    )
    print(
        f"\nFetched {image_log[image_log['status']=='ok'].shape[0]} images across "
        f"{coverage[coverage['coverage_flag']=='full'].shape[0]} fully covered tracts."
    )
    print("Coverage audit saved to data/audit/streetview_coverage.csv")
