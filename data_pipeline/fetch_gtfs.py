"""
fetch_gtfs.py — Download CTA + Metra GTFS transit schedules for Chicago.

GTFS (General Transit Feed Specification) data is public and contains:
- stops.txt: station/stop locations (lat/lon)
- routes.txt: transit lines (L, bus, Metra)
- stop_times.txt: scheduled arrivals (used for frequency/reliability analysis)
- trips.txt: trip → route mapping

We use GTFS to compute, per Census tract:
- nearest_station_dist_km: Euclidean distance to nearest transit stop
- travel_time_to_loop_min: estimated GTFS-based travel time to the Chicago Loop
- n_stops_within_500m: stop density (transit accessibility index)

Data sources (publicly available, no key needed):
- CTA: https://www.transitchicago.com/developers/gtfs.aspx
- Metra: https://metrarail.com/developers

Usage:
    python fetch_gtfs.py --output data/raw/gtfs/
"""

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger
from shapely.geometry import Point
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Public GTFS URLs ────────────────────────────────────────────────────────────

GTFS_SOURCES = {
    "cta": "https://www.transitchicago.com/downloads/sch_data/google_transit.zip",
    "metra": "https://www.metrarail.com/content/dam/metra/documents/GTFS.zip",
}

# Chicago Loop centroid (geographic center of downtown)
LOOP_LAT = 41.8827
LOOP_LON = -87.6278

# Walking speed for distance → time conversion when GTFS trips unavailable
WALK_SPEED_KMH = 5.0


# ── Download helpers ────────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
def _download_gtfs_zip(url: str) -> dict[str, pd.DataFrame]:
    """
    Download a GTFS zip from URL and parse core tables into DataFrames.

    Returns dict of {filename_stem: DataFrame} for:
    stops, routes, trips, stop_times, calendar, shapes (if present)
    """
    logger.info(f"Downloading GTFS from {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    tables = {}
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".txt"):
                stem = name.replace(".txt", "")
                try:
                    with zf.open(name) as f:
                        tables[stem] = pd.read_csv(f, dtype=str, low_memory=False)
                except Exception as e:
                    logger.warning(f"Could not parse {name}: {e}")

    required = {"stops", "routes", "trips", "stop_times"}
    missing = required - set(tables.keys())
    if missing:
        raise ValueError(f"GTFS zip missing required tables: {missing}")

    logger.info(f"GTFS parsed: {list(tables.keys())}")
    return tables


def _validate_stops(stops_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean GTFS stops table.

    Required columns: stop_id, stop_lat, stop_lon.
    Drops rows with missing/invalid coordinates.
    """
    stops_df = stops_df.copy()
    stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
    stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")
    n_before = len(stops_df)
    stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])
    # Basic bounds check (Chicago area)
    stops_df = stops_df[
        (stops_df["stop_lat"].between(41.5, 42.1)) & (stops_df["stop_lon"].between(-88.0, -87.4))
    ]
    n_after = len(stops_df)
    if n_before != n_after:
        logger.warning(f"Dropped {n_before - n_after} stops with invalid coordinates.")
    return stops_df


def _stops_to_geodataframe(stops_df: pd.DataFrame, agency: str) -> gpd.GeoDataFrame:
    """Convert stops DataFrame to GeoDataFrame with agency label."""
    stops_df = _validate_stops(stops_df)
    geometry = [Point(lon, lat) for lat, lon in zip(stops_df["stop_lat"], stops_df["stop_lon"])]
    gdf = gpd.GeoDataFrame(stops_df, geometry=geometry, crs="EPSG:4326")
    gdf["agency"] = agency
    cols = ["stop_id", "stop_name", "stop_lat", "stop_lon", "agency", "geometry"]
    cols = [c for c in cols if c in gdf.columns]
    return gdf[cols]


# ── Travel time estimation ──────────────────────────────────────────────────────


def estimate_travel_times_to_loop(
    stops_gdf: gpd.GeoDataFrame,
    stop_times_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate travel time (minutes) from each stop to the Chicago Loop.

    Method:
    1. For each stop, find all trip_ids serving that stop.
    2. For each trip, find the stop_time at the Loop-adjacent stop.
    3. Compute time difference = arrival_at_loop - departure_from_stop.
    4. Aggregate: median travel time per stop.

    Returns DataFrame: stop_id → estimated_travel_time_min.

    NOTE: This is an approximation. For precise times, use RAPTOR algorithm
    on full stop_times table (out of scope for Week 1).
    """
    # Identify Loop-area stops (within ~500m of Loop centroid)
    loop_pt = Point(LOOP_LON, LOOP_LAT)
    stops_proj = stops_gdf.to_crs(epsg=32616)  # UTM Zone 16N for meters
    loop_pt_proj = gpd.GeoSeries([loop_pt], crs="EPSG:4326").to_crs(epsg=32616).iloc[0]
    stops_proj["dist_to_loop_m"] = stops_proj.geometry.distance(loop_pt_proj)
    loop_stop_ids = set(stops_proj[stops_proj["dist_to_loop_m"] < 600]["stop_id"].tolist())

    if not loop_stop_ids:
        logger.warning("No stops found near Chicago Loop — using Euclidean fallback.")
        return pd.DataFrame(columns=["stop_id", "travel_time_to_loop_min"])

    logger.info(f"Found {len(loop_stop_ids)} Loop-area stops for travel time estimation.")

    # Parse arrival times (handle times past midnight: "25:30:00" etc.)
    def parse_gtfs_time(t: str) -> float:
        """Convert GTFS time string to minutes since midnight."""
        try:
            parts = str(t).strip().split(":")
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
        except Exception:
            return float("nan")

    stop_times = stop_times_df.copy()
    stop_times["departure_min"] = stop_times["departure_time"].apply(parse_gtfs_time)
    stop_times["arrival_min"] = stop_times["arrival_time"].apply(parse_gtfs_time)

    # For each trip: find min arrival at Loop stop
    loop_arrivals = (
        stop_times[stop_times["stop_id"].isin(loop_stop_ids)]
        .groupby("trip_id")["arrival_min"]
        .min()
        .reset_index()
        .rename(columns={"arrival_min": "loop_arrival_min"})
    )

    # For each non-Loop stop: find departure time on same trips
    non_loop_stops = stop_times[~stop_times["stop_id"].isin(loop_stop_ids)].copy()
    merged = non_loop_stops.merge(loop_arrivals, on="trip_id")
    merged["travel_time_min"] = merged["loop_arrival_min"] - merged["departure_min"]
    # Keep only forward-in-time trips (positive travel time, <120 min)
    merged = merged[(merged["travel_time_min"] > 0) & (merged["travel_time_min"] < 120)]

    travel_times = merged.groupby("stop_id")["travel_time_min"].median().reset_index()
    logger.info(f"Estimated travel times for {len(travel_times)} stops.")
    return travel_times


# ── Main ────────────────────────────────────────────────────────────────────────


def fetch_all_gtfs(output_dir: str | Path = "data/raw/gtfs/") -> dict[str, Path]:
    """
    Download and save CTA + Metra GTFS data.

    Returns paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stops = []
    paths = {}

    for agency, url in GTFS_SOURCES.items():
        agency_dir = output_dir / agency
        stops_path = agency_dir / "stops.geojson"
        tt_path = agency_dir / "travel_times_to_loop.csv"

        if stops_path.exists() and tt_path.exists():
            logger.info(f"{agency.upper()} GTFS already fetched — skipping.")
            paths[f"{agency}_stops"] = stops_path
            paths[f"{agency}_travel_times"] = tt_path
            all_stops.append(gpd.read_file(stops_path))
            continue

        agency_dir.mkdir(exist_ok=True)
        try:
            tables = _download_gtfs_zip(url)
        except Exception as e:
            logger.error(f"Failed to download {agency.upper()} GTFS: {e}")
            logger.warning(f"Skipping {agency.upper()} — will affect transit coverage.")
            continue

        # Save raw CSVs for reproducibility
        for table_name, df in tables.items():
            df.to_csv(agency_dir / f"{table_name}.csv", index=False)

        # Build stops GeoDataFrame
        stops_gdf = _stops_to_geodataframe(tables["stops"], agency=agency)
        stops_gdf.to_file(stops_path, driver="GeoJSON")
        paths[f"{agency}_stops"] = stops_path
        all_stops.append(stops_gdf)

        # Estimate travel times to Loop
        try:
            travel_times_df = estimate_travel_times_to_loop(
                stops_gdf=stops_gdf,
                stop_times_df=tables["stop_times"],
                trips_df=tables["trips"],
                routes_df=tables["routes"],
            )
            travel_times_df.to_csv(tt_path, index=False)
            paths[f"{agency}_travel_times"] = tt_path
        except Exception as e:
            logger.warning(f"Travel time estimation failed for {agency}: {e}")

        logger.info(f"{agency.upper()}: {len(stops_gdf)} stops saved → {stops_path}")

    # Merge CTA + Metra into unified stops file
    if all_stops:
        merged_stops = gpd.GeoDataFrame(pd.concat(all_stops, ignore_index=True), crs="EPSG:4326")
        merged_path = output_dir / "all_transit_stops.geojson"
        merged_stops.to_file(merged_path, driver="GeoJSON")
        paths["all_stops"] = merged_path
        logger.info(f"Merged transit stops: {len(merged_stops)} total → {merged_path}")

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Chicago CTA + Metra GTFS data")
    parser.add_argument("--output", default="data/raw/gtfs/", help="Output directory")
    args = parser.parse_args()

    paths = fetch_all_gtfs(output_dir=args.output)
    for name, path in paths.items():
        print(f"  {name:35s} → {path}")
