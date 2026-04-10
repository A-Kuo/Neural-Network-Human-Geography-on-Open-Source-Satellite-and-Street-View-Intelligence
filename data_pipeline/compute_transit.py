"""
compute_transit.py — Compute transit accessibility metrics per Census tract.

This is the core geographic feature for our research question:
"Can deep GNNs learn transit-mediated urban segregation patterns
that shallow networks systematically fail to capture?"

Transit accessibility is inherently multi-scale:
- Scale 1 (local): nearest stop within walking distance (~500m)
- Scale 2 (neighborhood): number of lines + routes accessible within 15-min walk
- Scale 3 (city): travel time to economic core (Loop)

This hierarchy is EXACTLY why depth matters:
- Shallow GNN: learns Scale 1 only (immediate neighbors)
- Deep GNN: can propagate Scale 3 signals through the transit graph

Inputs:
    data/raw/gtfs/all_transit_stops.geojson
    data/raw/gtfs/cta/travel_times_to_loop.csv   (from fetch_gtfs.py)
    data/raw/gtfs/metra/travel_times_to_loop.csv
    data/raw/census/cook_county_tracts_<year>.geojson

Output:
    data/processed/tract_transit_features.parquet

Metrics (all tract-level aggregates):
    nearest_stop_dist_km    — distance from tract centroid to nearest stop
    n_stops_500m            — stop density within 500m walking radius
    n_stops_1km             — stop density within 1km radius
    n_unique_routes_1km     — number of distinct routes accessible within 1km
    has_rail_access         — boolean: any L/Metra within 1km?
    median_travel_time_loop — median GTFS travel time to Chicago Loop
    transit_score           — composite accessibility score (0-100, see below)

Transit score formula (0-100):
    transit_score = 40 * (1/nearest_stop_dist_km normalized) +
                    30 * (n_stops_500m / max_stops_500m) +
                    30 * (1 - median_travel_time_loop / 120)

Usage:
    python compute_transit.py \
        --stops data/raw/gtfs/all_transit_stops.geojson \
        --tracts data/raw/census/cook_county_tracts_2022.geojson \
        --output data/processed/tract_transit_features.parquet
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger


# ── Radius thresholds for multi-scale features ──────────────────────────────────

RADIUS_500M = 500    # meters — reasonable walking distance
RADIUS_1KM = 1000   # meters — extended walking / bike distance

# Rail stop types in OSM/GTFS (for has_rail_access feature)
RAIL_STOP_TYPES = {
    "subway_entrance", "station", "halt", "platform",
    "stop_position",   # Metra/CTA heavy rail
}

# Travel time cap for normalization (2 hours max)
MAX_TRAVEL_TIME_MIN = 120.0

# ── Geometry helpers ─────────────────────────────────────────────────────────────

def _project_to_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project GeoDataFrame to UTM Zone 16N (EPSG:32616) for meter-accurate distances."""
    return gdf.to_crs(epsg=32616)


def compute_tract_centroids(tracts_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of tract centroids (projected to UTM for distance ops)."""
    tracts_proj = _project_to_utm(tracts_gdf[["tract_id", "geometry"]])
    centroids = tracts_proj.copy()
    centroids["geometry"] = tracts_proj.geometry.centroid
    return centroids


# ── Core spatial computations ────────────────────────────────────────────────────

def nearest_stop_distances(
    centroids_proj: gpd.GeoDataFrame,
    stops_proj: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Compute distance (km) from each tract centroid to its nearest transit stop.

    Uses spatial index (R-tree) for efficiency across ~2000 tracts × ~15000 stops.
    """
    from shapely.ops import nearest_points

    stop_sindex = stops_proj.sindex
    records = []

    for _, row in centroids_proj.iterrows():
        centroid = row.geometry
        # Query spatial index for candidate nearest stops
        candidate_idxs = list(stop_sindex.nearest(centroid, num_results=1))
        if not candidate_idxs:
            records.append({"tract_id": row["tract_id"], "nearest_stop_dist_km": np.nan,
                             "nearest_stop_type": None})
            continue

        nearest_stop = stops_proj.iloc[candidate_idxs[0]]
        dist_m = centroid.distance(nearest_stop.geometry)
        records.append({
            "tract_id": row["tract_id"],
            "nearest_stop_dist_km": dist_m / 1000.0,
            "nearest_stop_type": nearest_stop.get("stop_type", "unknown"),
        })

    return pd.DataFrame(records)


def stops_within_radius(
    centroids_proj: gpd.GeoDataFrame,
    stops_proj: gpd.GeoDataFrame,
    radius_m: float,
) -> pd.DataFrame:
    """
    Count stops within radius_m of each tract centroid.

    Also counts unique routes and whether any rail stop is present.
    """
    stop_sindex = stops_proj.sindex
    records = []

    for _, row in centroids_proj.iterrows():
        centroid = row.geometry
        # Buffer centroid, query spatial index
        buffer = centroid.buffer(radius_m)
        candidate_idxs = list(stop_sindex.intersection(buffer.bounds))
        nearby = stops_proj.iloc[candidate_idxs]
        # Exact intersection (spatial index gives bounding box candidates)
        nearby = nearby[nearby.geometry.within(buffer)]

        n_stops = len(nearby)
        has_rail = False
        if "stop_type" in nearby.columns:
            has_rail = any(
                str(t).lower() in RAIL_STOP_TYPES
                for t in nearby["stop_type"].dropna()
            )

        records.append({
            "tract_id": row["tract_id"],
            f"n_stops_{int(radius_m)}m": n_stops,
            "has_rail_access": has_rail,
        })

    return pd.DataFrame(records)


# ── Travel time aggregation ─────────────────────────────────────────────────────

def assign_travel_times_to_tracts(
    centroids_proj: gpd.GeoDataFrame,
    stops_proj: gpd.GeoDataFrame,
    travel_times_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each tract, find the nearest stop WITH a known travel time to the Loop,
    and assign that travel time to the tract.

    This is a conservative estimate: it's the best-case transit time to downtown,
    not accounting for walking time to the stop.
    """
    # Merge stops with travel times
    stops_with_tt = stops_proj.merge(travel_times_df, on="stop_id", how="left")
    stops_with_tt_valid = stops_with_tt.dropna(subset=["travel_time_to_loop_min"])

    if stops_with_tt_valid.empty:
        logger.warning("No valid travel times available. Using Euclidean distance fallback.")
        return pd.DataFrame({"tract_id": centroids_proj["tract_id"],
                             "median_travel_time_loop": np.nan})

    stop_sindex = stops_with_tt_valid.sindex
    records = []

    for _, row in centroids_proj.iterrows():
        centroid = row.geometry
        # Look within 2km radius for stops with known travel times
        buffer = centroid.buffer(2000)
        candidate_idxs = list(stop_sindex.intersection(buffer.bounds))
        nearby = stops_with_tt_valid.iloc[candidate_idxs]
        nearby = nearby[nearby.geometry.within(buffer)]

        if nearby.empty:
            # Expand to 5km if nothing nearby
            buffer5 = centroid.buffer(5000)
            candidate_idxs = list(stop_sindex.intersection(buffer5.bounds))
            nearby = stops_with_tt_valid.iloc[candidate_idxs]
            nearby = nearby[nearby.geometry.within(buffer5)]

        if nearby.empty:
            records.append({
                "tract_id": row["tract_id"],
                "median_travel_time_loop": np.nan,
                "min_travel_time_loop": np.nan,
            })
        else:
            records.append({
                "tract_id": row["tract_id"],
                "median_travel_time_loop": nearby["travel_time_to_loop_min"].median(),
                "min_travel_time_loop": nearby["travel_time_to_loop_min"].min(),
            })

    return pd.DataFrame(records)


# ── Composite transit score ──────────────────────────────────────────────────────

def compute_transit_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute composite transit accessibility score (0-100).

    Higher = better transit access.

    Components (weighted):
    - 40 pts: proximity to nearest stop (closer = higher)
    - 30 pts: stop density within 500m
    - 30 pts: travel time to Loop (faster = higher)

    This score is NOT used directly as a label — it's an input feature.
    The GNN must LEARN which combination of these scales predicts economic outcomes.
    """
    def normalize_inverse(series: pd.Series, cap: float) -> pd.Series:
        """Normalize inverse (lower distance = higher score), cap at cap."""
        clipped = series.clip(upper=cap)
        return 1.0 - (clipped / cap)

    def normalize(series: pd.Series) -> pd.Series:
        """Min-max normalize to [0, 1]."""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    score = pd.Series(0.0, index=df.index)

    if "nearest_stop_dist_km" in df.columns:
        score += 40 * normalize_inverse(df["nearest_stop_dist_km"].fillna(5.0), cap=3.0)

    if "n_stops_500m" in df.columns:
        score += 30 * normalize(df["n_stops_500m"].fillna(0))

    if "median_travel_time_loop" in df.columns:
        tt = df["median_travel_time_loop"].fillna(MAX_TRAVEL_TIME_MIN)
        score += 30 * normalize_inverse(tt, cap=MAX_TRAVEL_TIME_MIN)

    return score.clip(0, 100)


# ── Bias audit ──────────────────────────────────────────────────────────────────

def transit_coverage_audit(
    tract_transit: pd.DataFrame,
    income_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Document systematic transit coverage bias.

    Known issue: CTA covers inner-city Chicago well; Metra covers suburbs.
    South Side and West Side tracts historically underserved.
    If income_df provided: check whether low-income tracts have worse transit access.
    """
    audit = {
        "n_tracts_no_rail": (~tract_transit["has_rail_access"]).sum(),
        "n_tracts_missing_travel_time": tract_transit["median_travel_time_loop"].isna().sum(),
        "mean_nearest_stop_km": tract_transit["nearest_stop_dist_km"].mean(),
        "pct_tracts_within_500m": (tract_transit["n_stops_500m"] > 0).mean() * 100,
    }
    for k, v in audit.items():
        logger.info(f"  Transit audit — {k}: {v:.2f}" if isinstance(v, float) else
                    f"  Transit audit — {k}: {v}")

    if income_df is not None:
        merged = tract_transit.merge(
            income_df[["tract_id", "median_household_income"]], on="tract_id", how="inner"
        ).dropna(subset=["median_household_income", "transit_score"])
        if len(merged) > 10:
            corr = merged[["transit_score", "median_household_income"]].corr().iloc[0, 1]
            logger.info(f"  Transit score × income correlation: r={corr:.3f}")
            if corr > 0.3:
                logger.warning(
                    f"  BIAS FLAG: transit_score is correlated with income (r={corr:.3f}). "
                    "This is the key geographic pattern we're studying — document it explicitly."
                )

    return pd.DataFrame([audit])


# ── Main ────────────────────────────────────────────────────────────────────────

def compute_all(
    stops_path: str | Path,
    tracts_path: str | Path,
    output_path: str | Path = "data/processed/tract_transit_features.parquet",
    cta_travel_times_path: Optional[str | Path] = None,
    metra_travel_times_path: Optional[str | Path] = None,
    income_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Full transit accessibility computation pipeline.
    """
    stops_path = Path(stops_path)
    tracts_path = Path(tracts_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading transit stops: {stops_path}")
    stops = gpd.read_file(stops_path)
    stops_proj = _project_to_utm(stops)

    logger.info(f"Loading Census tracts: {tracts_path}")
    tracts = gpd.read_file(tracts_path)
    centroids = compute_tract_centroids(tracts)

    logger.info(f"Computing nearest stop distances for {len(centroids)} tracts...")
    nearest_df = nearest_stop_distances(centroids, stops_proj)

    logger.info(f"Counting stops within {RADIUS_500M}m radius...")
    stops_500m = stops_within_radius(centroids, stops_proj, RADIUS_500M)

    logger.info(f"Counting stops within {RADIUS_1KM}m radius...")
    stops_1km = stops_within_radius(centroids, stops_proj, RADIUS_1KM)
    stops_1km = stops_1km.rename(columns={"has_rail_access": "has_rail_access_1km"})

    # Merge travel times (CTA + Metra)
    travel_times_parts = []
    for tt_path in [cta_travel_times_path, metra_travel_times_path]:
        if tt_path and Path(tt_path).exists():
            travel_times_parts.append(pd.read_csv(tt_path))

    if travel_times_parts:
        all_tt = pd.concat(travel_times_parts, ignore_index=True)
        # If stop appears in both, take minimum travel time
        all_tt = all_tt.groupby("stop_id")["travel_time_to_loop_min"].min().reset_index()
        logger.info(f"Travel times available for {len(all_tt)} stops.")
        travel_time_df = assign_travel_times_to_tracts(centroids, stops_proj, all_tt)
    else:
        logger.warning("No GTFS travel time data found. travel_time_loop will be NaN.")
        travel_time_df = pd.DataFrame({
            "tract_id": centroids["tract_id"],
            "median_travel_time_loop": np.nan,
            "min_travel_time_loop": np.nan,
        })

    # Join all features
    result = (
        nearest_df
        .merge(stops_500m, on="tract_id", how="outer")
        .merge(stops_1km[["tract_id", f"n_stops_{RADIUS_1KM}m", "has_rail_access_1km"]],
               on="tract_id", how="outer")
        .merge(travel_time_df, on="tract_id", how="outer")
    )

    # Rename for clarity
    result = result.rename(columns={
        f"n_stops_{RADIUS_500M}m": "n_stops_500m",
        f"n_stops_{RADIUS_1KM}m": "n_stops_1km",
        "has_rail_access": "has_rail_500m",
        "has_rail_access_1km": "has_rail_1km",
    })

    # Composite score
    result["transit_score"] = compute_transit_score(result)

    # Bias audit
    income_df = None
    if income_path and Path(income_path).exists():
        income_df = pd.read_csv(income_path)
    audit_df = transit_coverage_audit(result, income_df)
    audit_dir = Path("data/audit")
    audit_dir.mkdir(exist_ok=True)
    audit_df.to_csv(audit_dir / "transit_coverage_audit.csv", index=False)

    result.to_parquet(output_path, index=False)
    logger.info(f"Transit features saved → {output_path}")
    logger.info(f"  {len(result)} tracts × {len(result.columns)} features")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute tract-level transit accessibility from GTFS + OSM stops"
    )
    parser.add_argument("--stops", default="data/raw/gtfs/all_transit_stops.geojson")
    parser.add_argument("--tracts", default="data/raw/census/cook_county_tracts_2022.geojson")
    parser.add_argument("--output", default="data/processed/tract_transit_features.parquet")
    parser.add_argument("--cta-travel-times", default="data/raw/gtfs/cta/travel_times_to_loop.csv")
    parser.add_argument("--metra-travel-times", default="data/raw/gtfs/metra/travel_times_to_loop.csv")
    parser.add_argument("--income", default=None)
    args = parser.parse_args()

    result = compute_all(
        stops_path=args.stops,
        tracts_path=args.tracts,
        output_path=args.output,
        cta_travel_times_path=args.cta_travel_times,
        metra_travel_times_path=args.metra_travel_times,
        income_path=args.income,
    )
    print(result[["tract_id", "nearest_stop_dist_km", "n_stops_500m",
                  "has_rail_500m", "median_travel_time_loop",
                  "transit_score"]].head(10).to_string())
