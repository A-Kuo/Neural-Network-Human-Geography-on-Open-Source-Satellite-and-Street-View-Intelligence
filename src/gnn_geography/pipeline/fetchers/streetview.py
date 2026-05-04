"""Google Street View image fetcher with rate limiting and coverage audit."""

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from tenacity import retry, stop_after_attempt, wait_exponential

from ...config import Config
from .base import BaseFetcher

# API endpoints
STREET_VIEW_API_URL = "https://maps.googleapis.com/maps/api/streetview"
METADATA_API_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"


@dataclass
class ImageRecord:
    """Metadata for one fetched Street View image. No lat/lon stored."""

    tract_id: str
    image_index: int
    heading: int
    filename: str
    fetch_date: str
    status: str  # "ok" | "no_imagery" | "error"
    pano_id: Optional[str] = None


@dataclass
class TractCoverageRecord:
    """Coverage audit record per Census tract."""

    tract_id: str
    n_requested: int
    n_fetched: int
    n_no_imagery: int
    n_errors: int
    coverage_flag: str  # "full" | "sparse" | "missing"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def check_imagery_available(lat: float, lon: float, api_key: str, radius_m: int = 50) -> Optional[str]:
    """Check if Street View imagery exists near (lat, lon)."""
    params = {
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "key": api_key,
    }
    resp = requests.get(METADATA_API_URL, params=params, timeout=15)
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
    fov: int = 90,
    pitch: int = 0,
    img_size: str = "640x640",
) -> bool:
    """Fetch one Street View image. Returns True on success."""
    params = {
        "size": img_size,
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": api_key,
        "return_error_code": True,
    }
    resp = requests.get(STREET_VIEW_API_URL, params=params, timeout=15)
    if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
        output_path.write_bytes(resp.content)
        return True
    return False


def sample_points_in_tract(
    tract_geometry,
    n_points: int = 8,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Generate up to n_points sampling locations within a Census tract polygon."""
    bounds = tract_geometry.bounds
    rng = np.random.default_rng(seed)

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


class StreetViewFetcher(BaseFetcher):
    """Fetcher for Google Street View images."""

    def __init__(self, config: Config):
        """Initialize Street View fetcher.

        Args:
            config: Configuration instance with Street View settings
        """
        super().__init__(config, "streetview")

    def fetch(self) -> dict[str, Path]:
        """Fetch Street View images for all Census tracts.

        Returns:
            Dictionary mapping output names to file paths
        """
        sv_config = self.config.streetview
        tracts_path = Path(self.config.paths.tract_centroids)
        output_dir = Path(self.config.paths.streetview_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get API key
        api_key = os.environ.get("GOOGLE_STREETVIEW_API_KEY", "")
        if not api_key:
            self.logger.warning("GOOGLE_STREETVIEW_API_KEY not set; skipping Street View fetch")
            return {}

        if not tracts_path.exists():
            self.logger.error(f"Tract centroids file not found: {tracts_path}")
            return {}

        tracts = gpd.read_file(tracts_path)
        if "tract_id" not in tracts.columns:
            raise ValueError("GeoJSON must have 'tract_id' column")

        # Ensure WGS84
        tracts = tracts.to_crs(epsg=4326)

        n_points_per_tract = max(1, sv_config.images_per_tract_target // len(sv_config.headings))
        image_records: list[ImageRecord] = []
        coverage_records: list[TractCoverageRecord] = []

        self.logger.info(
            f"Fetching Street View for {len(tracts)} tracts "
            f"({sv_config.images_per_tract_target} images/tract target)"
        )

        for _, row in tracts.iterrows():
            tract_id = str(row["tract_id"])
            tract_dir = output_dir / tract_id

            tract_dir.mkdir(exist_ok=True)
            sampling_points = sample_points_in_tract(row.geometry, n_points=n_points_per_tract)

            n_fetched = 0
            n_no_imagery = 0
            n_errors = 0
            img_index = 0

            for lat, lon in sampling_points:
                # Check availability first (free metadata endpoint)
                try:
                    pano_id = check_imagery_available(
                        lat, lon, api_key, radius_m=sv_config.location_radius_m
                    )
                except Exception as e:
                    self.logger.warning(f"Metadata check failed at ({lat:.4f},{lon:.4f}): {e}")
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

                for heading in sv_config.headings:
                    filename = f"{tract_id}_{img_index:04d}_h{heading}.jpg"
                    filepath = tract_dir / filename

                    # Rate limiting: 1 request per second
                    time.sleep(1.0 / sv_config.requests_per_second)

                    try:
                        ok = fetch_single_image(
                            lat, lon, heading, api_key, filepath,
                            fov=sv_config.fov,
                            pitch=sv_config.pitch,
                            img_size=sv_config.image_size,
                        )
                        status = "ok" if ok else "error"
                        if ok:
                            n_fetched += 1
                        else:
                            n_errors += 1
                    except Exception as e:
                        self.logger.error(f"Fetch failed {filename}: {e}")
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

            # Coverage classification
            if n_fetched == 0:
                flag = "missing"
            elif n_fetched < sv_config.min_images_for_coverage:
                flag = "sparse"
            else:
                flag = "full"

            coverage_records.append(
                TractCoverageRecord(
                    tract_id=tract_id,
                    n_requested=len(sampling_points) * len(sv_config.headings),
                    n_fetched=n_fetched,
                    n_no_imagery=n_no_imagery,
                    n_errors=n_errors,
                    coverage_flag=flag,
                )
            )
            self.logger.info(f"Tract {tract_id}: {n_fetched} images ({flag})")

        # Save logs
        image_log_df = pd.DataFrame([asdict(r) for r in image_records])
        coverage_df = pd.DataFrame([asdict(r) for r in coverage_records])

        audit_dir = Path(self.config.paths.audit)
        audit_dir.mkdir(exist_ok=True)
        image_log_path = audit_dir / "streetview_image_log.csv"
        coverage_path = audit_dir / "streetview_coverage.csv"
        image_log_df.to_csv(image_log_path, index=False)
        coverage_df.to_csv(coverage_path, index=False)

        self._log_coverage_summary(coverage_df, sv_config.min_images_for_coverage)

        return {
            "image_log": image_log_path,
            "coverage": coverage_path,
        }

    def _log_coverage_summary(self, coverage_df: pd.DataFrame, min_images: int) -> None:
        """Log coverage statistics."""
        total = len(coverage_df)
        full = (coverage_df["coverage_flag"] == "full").sum()
        sparse = (coverage_df["coverage_flag"] == "sparse").sum()
        missing = (coverage_df["coverage_flag"] == "missing").sum()

        self.logger.info("=" * 60)
        self.logger.info("STREET VIEW COVERAGE AUDIT")
        self.logger.info(f"  Total tracts: {total}")
        self.logger.info(f"  Full coverage (>={min_images} imgs): {full} ({100*full/total:.1f}%)")
        self.logger.info(f"  Sparse (<{min_images} imgs):         {sparse} ({100*sparse/total:.1f}%)")
        self.logger.info(f"  Missing (0 imgs):                {missing} ({100*missing/total:.1f}%)")
        if missing > 0:
            missing_ids = coverage_df[coverage_df["coverage_flag"] == "missing"]["tract_id"].tolist()
            self.logger.warning(f"  Missing tract IDs: {missing_ids}")
        self.logger.info("=" * 60)
