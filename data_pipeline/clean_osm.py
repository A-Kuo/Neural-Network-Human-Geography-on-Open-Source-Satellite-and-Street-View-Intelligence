"""
clean_osm.py — Aggregate OSM building footprints to Census tract-level statistics.

Input:
    data/raw/osm_buildings.geojson   — individual building polygons
    data/raw/census/cook_county_tracts_<year>.geojson — tract boundaries

Output:
    data/processed/tract_building_stats.parquet

Metrics computed per tract (all aggregated — no individual building linked to income):
    building_count          — number of OSM buildings
    building_density_km2    — buildings per km² of land area
    median_building_area_m2 — median footprint area
    median_height_m         — median estimated building height
    pct_residential         — fraction of buildings tagged as residential
    pct_commercial          — fraction tagged as commercial/retail
    pct_missing_height      — proxy for OSM data completeness (bias indicator)

Usage:
    python clean_osm.py \
        --buildings data/raw/osm_buildings.geojson \
        --tracts data/raw/census/cook_county_tracts_2022.geojson \
        --output data/processed/tract_building_stats.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger


# ── Building type categorization ────────────────────────────────────────────────

RESIDENTIAL_TAGS = {
    "residential", "house", "detached", "apartments", "apartment",
    "dormitory", "terrace", "semidetached_house", "bungalow", "cabin",
}

COMMERCIAL_TAGS = {
    "commercial", "retail", "office", "supermarket", "shop",
    "warehouse", "industrial", "civic", "government",
}


def classify_building(tag: str) -> str:
    """Map OSM building tag to coarse category."""
    if not isinstance(tag, str):
        return "unknown"
    tag_lower = tag.lower()
    if tag_lower in RESIDENTIAL_TAGS:
        return "residential"
    if tag_lower in COMMERCIAL_TAGS:
        return "commercial"
    if tag_lower in ("yes", "building"):
        return "unspecified"
    return "other"


# ── Geometry processing ─────────────────────────────────────────────────────────

def compute_building_areas(buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute building footprint areas in m².

    Projects to UTM Zone 16N (EPSG:32616) for accurate area computation in meters.
    """
    buildings_proj = buildings_gdf.to_crs(epsg=32616)
    buildings_proj["area_m2"] = buildings_proj.geometry.area
    # Copy back the computed area; geometry stays in projected CRS for spatial join
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["area_m2"] = buildings_proj["area_m2"].values
    return buildings_gdf


def spatial_join_to_tracts(
    buildings_gdf: gpd.GeoDataFrame,
    tracts_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign each building to a Census tract via spatial join (centroid method).

    Using building centroid prevents double-counting for buildings on tract boundaries.
    """
    buildings_4326 = buildings_gdf.to_crs(epsg=4326)
    tracts_4326 = tracts_gdf[["tract_id", "geometry"]].to_crs(epsg=4326)

    # Use building centroid for join
    centroids = buildings_4326.copy()
    centroids["geometry"] = buildings_4326.geometry.centroid

    joined = gpd.sjoin(
        centroids,
        tracts_4326,
        how="left",
        predicate="within",
    )
    # Drop buildings outside any tract (county edge artifacts)
    n_before = len(joined)
    joined = joined.dropna(subset=["tract_id"])
    n_dropped = n_before - len(joined)
    if n_dropped:
        logger.info(f"Dropped {n_dropped} buildings outside Cook County tracts.")

    return joined


# ── Aggregation ─────────────────────────────────────────────────────────────────

def aggregate_to_tract(
    joined_gdf: gpd.GeoDataFrame,
    tracts_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Compute tract-level building statistics.

    Returns one row per tract (left join — tracts with zero buildings get NaN).
    """
    # Add land area for density computation
    tracts_proj = tracts_gdf.to_crs(epsg=32616)
    tracts_proj["land_area_km2"] = tracts_proj.geometry.area / 1e6  # m² → km²
    tract_areas = tracts_proj.set_index("tract_id")["land_area_km2"]

    # Classify building types
    joined_gdf = joined_gdf.copy()
    joined_gdf["building_category"] = joined_gdf["building_type"].apply(classify_building)

    def _agg(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        return pd.Series({
            "building_count": n,
            "median_building_area_m2": group["area_m2"].median(),
            "median_height_m": group["height_m"].median(),  # NaN if all missing
            "pct_residential": (group["building_category"] == "residential").sum() / n,
            "pct_commercial": (group["building_category"] == "commercial").sum() / n,
            "pct_missing_height": group["height_m"].isna().sum() / n,
            "mean_building_area_m2": group["area_m2"].mean(),
            "total_footprint_area_m2": group["area_m2"].sum(),
        })

    tract_stats = joined_gdf.groupby("tract_id").apply(_agg).reset_index()

    # Compute building density using land area
    tract_stats["land_area_km2"] = tract_stats["tract_id"].map(tract_areas)
    tract_stats["building_density_km2"] = (
        tract_stats["building_count"] / tract_stats["land_area_km2"]
    )

    # Left join to include tracts with zero buildings
    all_tracts = tracts_gdf[["tract_id"]].copy()
    result = all_tracts.merge(tract_stats, on="tract_id", how="left")

    # Fill zero for count (0 buildings is real, not missing)
    result["building_count"] = result["building_count"].fillna(0).astype(int)

    logger.info(f"Building stats aggregated: {len(result)} tracts")
    n_zero = (result["building_count"] == 0).sum()
    if n_zero:
        logger.warning(f"  {n_zero} tracts have 0 OSM buildings — possible sparse coverage.")

    return result


# ── Bias audit for OSM coverage ─────────────────────────────────────────────────

def osm_coverage_audit(
    tract_stats: pd.DataFrame,
    income_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assess OSM data quality and potential wealth-correlated coverage bias.

    Key concern: wealthy neighborhoods tend to have more detailed OSM data
    (more contributors, better tagging). This creates systematic bias.

    If income_df is provided, computes correlation between building completeness
    (pct_missing_height, building_density_km2) and income.
    """
    audit_rows = []
    metrics = [
        ("building_count", "OSM building count"),
        ("pct_missing_height", "% buildings missing height tag"),
        ("building_density_km2", "Building density (bldgs/km²)"),
    ]
    for col, label in metrics:
        if col not in tract_stats.columns:
            continue
        audit_rows.append({
            "metric": label,
            "mean": tract_stats[col].mean(),
            "std": tract_stats[col].std(),
            "pct_missing": 100 * tract_stats[col].isna().mean(),
        })

    audit_df = pd.DataFrame(audit_rows)

    if income_df is not None:
        merged = tract_stats.merge(income_df[["tract_id", "median_household_income"]],
                                   on="tract_id", how="inner")
        merged = merged.dropna(subset=["median_household_income"])
        for col, label in metrics:
            if col not in merged.columns:
                continue
            valid = merged[[col, "median_household_income"]].dropna()
            if len(valid) > 10:
                corr = valid.corr().iloc[0, 1]
                logger.info(f"  OSM bias check — {label} × income: r={corr:.3f}")
                # Flag if strongly correlated (possible wealth-based data density bias)
                if abs(corr) > 0.3:
                    logger.warning(f"  BIAS FLAG: {label} correlated with income (r={corr:.3f}). "
                                   f"Document in ETHICS.md.")

    return audit_df


# ── Main ────────────────────────────────────────────────────────────────────────

# Allow optional income_df for audit
from typing import Optional


def clean_all(
    buildings_path: str | Path,
    tracts_path: str | Path,
    output_path: str | Path = "data/processed/tract_building_stats.parquet",
    income_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Full OSM cleaning pipeline: load → area computation → spatial join → aggregate.
    """
    buildings_path = Path(buildings_path)
    tracts_path = Path(tracts_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading buildings: {buildings_path}")
    buildings = gpd.read_file(buildings_path)
    logger.info(f"  {len(buildings)} building polygons loaded.")

    logger.info(f"Loading tracts: {tracts_path}")
    tracts = gpd.read_file(tracts_path)
    logger.info(f"  {len(tracts)} Census tracts loaded.")

    logger.info("Computing building footprint areas...")
    buildings = compute_building_areas(buildings)

    logger.info("Spatial join: buildings → tracts...")
    joined = spatial_join_to_tracts(buildings, tracts)

    logger.info("Aggregating to tract level...")
    tract_stats = aggregate_to_tract(joined, tracts)

    # Bias audit
    income_df = None
    if income_path and Path(income_path).exists():
        income_df = pd.read_csv(income_path)
    audit_df = osm_coverage_audit(tract_stats, income_df)
    audit_dir = Path("data/audit")
    audit_dir.mkdir(exist_ok=True)
    audit_df.to_csv(audit_dir / "osm_coverage_audit.csv", index=False)

    tract_stats.to_parquet(output_path, index=False)
    logger.info(f"Building stats saved → {output_path}")
    return tract_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean OSM buildings → tract-level stats")
    parser.add_argument("--buildings", default="data/raw/osm_buildings.geojson")
    parser.add_argument("--tracts", default="data/raw/census/cook_county_tracts_2022.geojson")
    parser.add_argument("--output", default="data/processed/tract_building_stats.parquet")
    parser.add_argument("--income", default=None,
                        help="Optional ACS CSV for bias correlation check")
    args = parser.parse_args()

    stats = clean_all(
        buildings_path=args.buildings,
        tracts_path=args.tracts,
        output_path=args.output,
        income_path=args.income,
    )
    print(f"\nDone. {len(stats)} tracts × {len(stats.columns)} features → {args.output}")
    print(stats[["tract_id", "building_count", "building_density_km2",
                 "median_height_m", "pct_residential"]].head(10).to_string())
