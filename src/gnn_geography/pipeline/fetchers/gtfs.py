"""GTFS transit data fetcher for CTA and Metra."""

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from tenacity import retry, stop_after_attempt, wait_exponential

from ...config import Config
from .base import BaseFetcher


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
def _download_gtfs_zip(url: str) -> dict[str, pd.DataFrame]:
    """Download a GTFS zip from URL and parse core tables into DataFrames."""
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
                    pass

    required = {"stops", "routes", "trips", "stop_times"}
    missing = required - set(tables.keys())
    if missing:
        raise ValueError(f"GTFS zip missing required tables: {missing}")

    return tables


def _validate_stops(stops_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean GTFS stops table."""
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


def estimate_travel_times_to_loop(
    stops_gdf: gpd.GeoDataFrame,
    stop_times_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    loop_lat: float,
    loop_lon: float,
    loop_radius_m: int = 600,
) -> pd.DataFrame:
    """Estimate travel time (minutes) from each stop to the Chicago Loop."""
    # Identify Loop-area stops
    loop_pt = Point(loop_lon, loop_lat)
    stops_proj = stops_gdf.to_crs(epsg=32616)
    loop_pt_proj = gpd.GeoSeries([loop_pt], crs="EPSG:4326").to_crs(epsg=32616).iloc[0]
    stops_proj["dist_to_loop_m"] = stops_proj.geometry.distance(loop_pt_proj)
    loop_stop_ids = set(stops_proj[stops_proj["dist_to_loop_m"] < loop_radius_m]["stop_id"].tolist())

    if not loop_stop_ids:
        return pd.DataFrame(columns=["stop_id", "travel_time_to_loop_min"])

    # Parse GTFS times (handle times past midnight)
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
    return travel_times


class GTFSFetcher(BaseFetcher):
    """Fetcher for GTFS transit schedule data (CTA + Metra)."""

    def __init__(self, config: Config):
        """Initialize GTFS fetcher.

        Args:
            config: Configuration instance with GTFS settings
        """
        super().__init__(config, "gtfs")

    def fetch(self) -> dict[str, Path]:
        """Fetch CTA + Metra GTFS data and compute travel times.

        Returns:
            Dictionary mapping output names to file paths
        """
        gtfs_config = self.config.gtfs
        output_dir = Path(self.config.paths.gtfs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_stops = []
        paths = {}

        for agency, url in gtfs_config.sources.items():
            agency_dir = output_dir / agency
            stops_path = agency_dir / "stops.geojson"
            tt_path = agency_dir / "travel_times_to_loop.csv"

            if stops_path.exists() and tt_path.exists():
                self.logger.info(f"{agency.upper()} GTFS already fetched")
                paths[f"{agency}_stops"] = stops_path
                paths[f"{agency}_travel_times"] = tt_path
                all_stops.append(gpd.read_file(stops_path))
                continue

            agency_dir.mkdir(exist_ok=True)
            try:
                tables = _download_gtfs_zip(url)
            except Exception as e:
                self.logger.error(f"Failed to download {agency.upper()} GTFS: {e}")
                continue

            # Save raw CSVs
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
                    loop_lat=self.config.region.loop_lat,
                    loop_lon=self.config.region.loop_lon,
                    loop_radius_m=gtfs_config.loop_stop_radius_m,
                )
                travel_times_df.to_csv(tt_path, index=False)
                paths[f"{agency}_travel_times"] = tt_path
            except Exception as e:
                self.logger.warning(f"Travel time estimation failed for {agency}: {e}")

            self.logger.info(f"{agency.upper()}: {len(stops_gdf)} stops saved → {stops_path}")

        # Merge CTA + Metra into unified stops file
        if all_stops:
            merged_stops = gpd.GeoDataFrame(
                pd.concat(all_stops, ignore_index=True), crs="EPSG:4326"
            )
            merged_path = output_dir / "all_transit_stops.geojson"
            merged_stops.to_file(merged_path, driver="GeoJSON")
            paths["all_stops"] = merged_path
            self.logger.info(f"Merged transit stops: {len(merged_stops)} total → {merged_path}")

        return paths
