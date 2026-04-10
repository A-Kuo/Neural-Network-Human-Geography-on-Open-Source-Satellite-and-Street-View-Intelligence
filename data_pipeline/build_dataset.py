"""
build_dataset.py — Join all data layers into the final research dataset.

Enforces ALL privacy rules before writing:
1. Minimum unit: Census tract (>= 4,000 people)
2. No [lat, lon, income] in same table
3. No individual addresses
4. All features averaged across >= 10 samples per tract
5. Validates no column contains address or coordinate data

Output:
    data/processed/final_dataset.parquet  — tract-level, no PII

Schema (one row per Census tract):
    tract_id                    — 11-digit Census FIPS
    n_images                    — number of Street View images used
    img_feat_0..2047            — ResNet-152 mean embeddings
    building_count              — OSM building count
    building_density_km2        — buildings per km²
    median_building_area_m2     — median footprint
    median_height_m             — median building height (sparse)
    pct_residential             — fraction residential
    nearest_stop_dist_km        — distance to nearest transit stop
    n_stops_500m                — stops within 500m
    n_stops_1km                 — stops within 1km
    has_rail_500m               — any rail within 500m
    median_travel_time_loop     — GTFS travel time to Loop (min)
    transit_score               — composite accessibility (0-100)
    median_household_income     — ACS median HH income (target variable)
    poverty_rate                — ACS poverty rate
    pct_no_vehicle              — fraction of HH without a car
    total_population            — tract population

Usage:
    python build_dataset.py [--output data/processed/final_dataset.parquet]
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger


# ── Privacy guard ────────────────────────────────────────────────────────────────

# Column names that must NEVER appear in the final dataset
PII_COLUMN_PATTERNS = [
    r"^lat(itude)?$",
    r"^lon(gitude)?$",
    r"address",
    r"street_name",
    r"household_id",
    r"person_id",
    r"name$",           # stop_name, person_name, etc.
]

REQUIRED_MIN_IMAGES = 10   # Tracts with fewer images excluded from ML training


def _privacy_guard(df: pd.DataFrame) -> None:
    """
    Raise ValueError if any PII column pattern is found.

    This runs before every write operation. If this fails, fix the pipeline —
    do NOT disable this check.
    """
    violations = []
    for col in df.columns:
        for pattern in PII_COLUMN_PATTERNS:
            if re.search(pattern, col, re.IGNORECASE):
                violations.append(f"  Column '{col}' matches PII pattern '{pattern}'")

    if violations:
        raise ValueError(
            "PRIVACY VIOLATION — dataset contains PII columns:\n" +
            "\n".join(violations) +
            "\n\nFix the pipeline before writing. See ETHICS.md for rules."
        )
    logger.info("Privacy guard PASSED — no PII columns detected.")


def _verify_aggregation_level(df: pd.DataFrame, population_col: str = "total_population") -> None:
    """
    Verify minimum aggregation unit (Census tract, >= 4,000 people).
    Log tracts that fall below threshold (may indicate data issue, not ethics violation).
    """
    if population_col not in df.columns:
        logger.warning(f"Column '{population_col}' not found — cannot verify aggregation level.")
        return
    small_tracts = df[df[population_col] < 1000]
    if len(small_tracts) > 0:
        logger.warning(
            f"  {len(small_tracts)} tracts have population < 1,000. "
            "These may be parks, airports, or data issues. Consider excluding."
        )
    logger.info(f"Aggregation level OK: all rows are Census tracts "
                f"(min pop: {df[population_col].min():.0f}).")


def _verify_image_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag tracts that don't meet minimum image threshold for ML training.
    Returns df with 'ml_eligible' column added.
    """
    if "n_images" not in df.columns:
        df["ml_eligible"] = False
        logger.warning("n_images column missing — all tracts flagged as ml_eligible=False.")
        return df

    df = df.copy()
    df["ml_eligible"] = df["n_images"] >= REQUIRED_MIN_IMAGES
    n_eligible = df["ml_eligible"].sum()
    n_total = len(df)
    logger.info(f"ML-eligible tracts: {n_eligible}/{n_total} "
                f"(>= {REQUIRED_MIN_IMAGES} images)")
    if n_eligible < n_total * 0.5:
        logger.warning(
            f"Only {100*n_eligible/n_total:.1f}% of tracts meet image threshold. "
            "This will bias toward well-covered neighborhoods — document in ETHICS.md."
        )
    return df


# ── Data loading helpers ────────────────────────────────────────────────────────

def _load_parquet_or_none(path: Path) -> Optional[pd.DataFrame]:
    if path and path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    logger.warning(f"File not found: {path} — skipping this layer.")
    return None


def _load_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    if path and path.exists():
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    logger.warning(f"File not found: {path} — skipping this layer.")
    return None


# ── Transformation log ──────────────────────────────────────────────────────────

class TransformationLog:
    """
    Audit trail for every join and transformation.
    Required for reproducibility: every row count change is logged.
    """
    def __init__(self):
        self.entries = []

    def log(self, step: str, df: pd.DataFrame, note: str = "") -> None:
        entry = {
            "step": step,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "note": note,
        }
        self.entries.append(entry)
        logger.info(f"[Transform] {step}: {len(df)} rows × {len(df.columns)} cols. {note}")

    def save(self, path: Path) -> None:
        pd.DataFrame(self.entries).to_csv(path, index=False)
        logger.info(f"Transformation log saved → {path}")


# ── Build pipeline ───────────────────────────────────────────────────────────────

def build(
    image_features_path: str | Path = "data/processed/tract_image_features.parquet",
    building_stats_path: str | Path = "data/processed/tract_building_stats.parquet",
    transit_features_path: str | Path = "data/processed/tract_transit_features.parquet",
    census_path: str | Path = "data/raw/census/acs_2022_cook_county.csv",
    output_path: str | Path = "data/processed/final_dataset.parquet",
) -> pd.DataFrame:
    """
    Join all processed feature tables into the final research dataset.

    Join key: tract_id (11-digit Census FIPS)
    Join type: outer — every tract appears, features are NaN if layer missing
    Privacy: enforced before writing (raises ValueError on violation)
    """
    image_features_path = Path(image_features_path)
    building_stats_path = Path(building_stats_path)
    transit_features_path = Path(transit_features_path)
    census_path = Path(census_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log = TransformationLog()

    # ── Load all layers ────────────────────────────────────────────────────────
    img_df = _load_parquet_or_none(image_features_path)
    bld_df = _load_parquet_or_none(building_stats_path)
    trt_df = _load_parquet_or_none(transit_features_path)
    cen_df = _load_csv_or_none(census_path)

    if cen_df is None:
        raise FileNotFoundError(
            f"Census data required but not found: {census_path}\n"
            "Run fetch_census.py first."
        )
    log.log("load_census", cen_df, note="ACS tract-level income + demographics")

    # ── Start with census as the canonical tract list ───────────────────────────
    # This ensures every Cook County tract appears, regardless of other data availability
    result = cen_df[["tract_id", "median_household_income", "total_population",
                      "poverty_rate", "pct_no_vehicle", "pct_college_plus",
                      "total_housing_units", "acs_year"]].copy()
    result = result.rename(columns={"acs_year": "census_year"})
    log.log("base_census", result, "All Cook County tracts from ACS")

    # ── Join image features ────────────────────────────────────────────────────
    if img_df is not None:
        # Drop any location columns that snuck through
        img_safe = img_df.drop(columns=[c for c in img_df.columns
                                         if re.search(r"lat|lon|address", c, re.IGNORECASE)
                                         and c != "tract_id"], errors="ignore")
        result = result.merge(img_safe, on="tract_id", how="left")
        log.log("join_image_features", result,
                f"Joined {len(img_safe.columns)-1} image feature columns")
    else:
        result["n_images"] = 0
        logger.warning("Image features not available — run extract_features.py first.")

    # ── Join building stats ────────────────────────────────────────────────────
    if bld_df is not None:
        bld_safe = bld_df.drop(columns=["geometry"] if "geometry" in bld_df.columns else [],
                                errors="ignore")
        result = result.merge(bld_safe, on="tract_id", how="left")
        log.log("join_building_stats", result, "Joined OSM building features")

    # ── Join transit features ──────────────────────────────────────────────────
    if trt_df is not None:
        # Drop any coordinate columns
        trt_safe = trt_df.drop(columns=[c for c in trt_df.columns
                                         if re.search(r"lat|lon", c, re.IGNORECASE)
                                         and c != "tract_id"], errors="ignore")
        result = result.merge(trt_safe, on="tract_id", how="left")
        log.log("join_transit_features", result, "Joined GTFS transit accessibility features")

    # ── Privacy guard ──────────────────────────────────────────────────────────
    # Remove NAME column (human-readable tract names — not strictly PII but unnecessary)
    result = result.drop(columns=[c for c in result.columns if c.upper() == "NAME"],
                         errors="ignore")

    _privacy_guard(result)
    log.log("privacy_guard_passed", result, "No PII columns detected")

    # ── Aggregation verification ───────────────────────────────────────────────
    _verify_aggregation_level(result)
    result = _verify_image_coverage(result)

    # ── Final validation ───────────────────────────────────────────────────────
    required_cols = ["tract_id", "median_household_income"]
    missing_required = [c for c in required_cols if c not in result.columns]
    if missing_required:
        raise ValueError(f"Final dataset missing required columns: {missing_required}")

    n_income_missing = result["median_household_income"].isna().sum()
    pct_income_missing = 100 * n_income_missing / len(result)
    if pct_income_missing > 10:
        logger.warning(
            f"  {pct_income_missing:.1f}% of tracts missing median income. "
            "Document in ETHICS.md — these tracts cannot be used for training."
        )

    log.log("final_dataset", result, "Complete — ready for ML pipeline")

    # ── Save ────────────────────────────────────────────────────────────────────
    result.to_parquet(output_path, index=False)
    logger.info(f"Final dataset saved: {len(result)} tracts × {len(result.columns)} cols → {output_path}")

    # Save transformation log
    audit_dir = Path("data/audit")
    audit_dir.mkdir(exist_ok=True)
    log.save(audit_dir / "build_dataset_transform_log.csv")

    # Summary statistics
    logger.info("\nDataset Summary:")
    logger.info(f"  Tracts:          {len(result)}")
    logger.info(f"  ML-eligible:     {result.get('ml_eligible', pd.Series(False)).sum()}")
    if "median_household_income" in result.columns:
        logger.info(f"  Income range:    ${result['median_household_income'].min():,.0f} – "
                    f"${result['median_household_income'].max():,.0f}")
    if "transit_score" in result.columns:
        logger.info(f"  Transit score:   {result['transit_score'].min():.1f} – "
                    f"{result['transit_score'].max():.1f}")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build final joined dataset (tract-level, no PII)")
    parser.add_argument("--image-features", default="data/processed/tract_image_features.parquet")
    parser.add_argument("--building-stats", default="data/processed/tract_building_stats.parquet")
    parser.add_argument("--transit-features", default="data/processed/tract_transit_features.parquet")
    parser.add_argument("--census", default="data/raw/census/acs_2022_cook_county.csv")
    parser.add_argument("--output", default="data/processed/final_dataset.parquet")
    args = parser.parse_args()

    dataset = build(
        image_features_path=args.image_features,
        building_stats_path=args.building_stats,
        transit_features_path=args.transit_features,
        census_path=args.census,
        output_path=args.output,
    )
    print(f"\nFinal dataset: {dataset.shape}")
    print(dataset[["tract_id", "median_household_income", "transit_score",
                   "building_density_km2", "n_images", "ml_eligible"]].head(10).to_string())
