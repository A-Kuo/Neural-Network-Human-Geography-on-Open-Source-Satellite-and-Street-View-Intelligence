"""
fetch_osm.py — Download Chicago building + street network data from OpenStreetMap.

Uses the Overpass API for building footprints and osmnx for the street/transit
graph. All data aggregated to Census tract level before joining income data.

Outputs:
    data/raw/osm_buildings.geojson    — building polygons with tract_id
    data/raw/osm_streets.graphml      — street network graph (Chicago bbox)
    data/raw/osm_transit_stops.geojson — bus/rail stop locations

Usage:
    python fetch_osm.py --bbox chicago [--output data/raw/]
"""

import time
import json
from pathlib import Path
from typing import Optional

import requests
import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import shape, mapping
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


# ── Chicago bounding box (WGS84) ───────────────────────────────────────────────

CHICAGO_BBOX = {
    "north": 42.023,
    "south": 41.644,
    "east":  -87.524,
    "west":  -87.940,
}

# Overpass API endpoint (public; no key needed)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 120  # seconds — large queries need time


# ── Overpass query helpers ──────────────────────────────────────────────────────

def _bbox_str(bbox: dict) -> str:
    """Format bbox for Overpass QL: (south, west, north, east)."""
    return f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}"


def _buildings_query(bbox: dict) -> str:
    """
    Overpass QL: fetch all building footprints in bbox.

    Returns ways + relations tagged as buildings. We request:
    - building=* (any building type)
    - Key fields: building, building:levels, height, amenity
    """
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["building"]({bb});
  relation["building"]({bb});
);
out body;
>;
out skel qt;
"""


def _transit_stops_query(bbox: dict) -> str:
    """
    Overpass QL: fetch CTA + Metra transit stops.

    Captures: subway stations, bus stops, rail platforms.
    """
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  node["railway"="station"]({bb});
  node["railway"="subway_entrance"]({bb});
  node["public_transport"="station"]({bb});
  node["public_transport"="stop_position"]({bb});
  node["highway"="bus_stop"]({bb});
);
out body;
"""


# ── Overpass fetcher ────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
def _overpass_fetch(query: str) -> dict:
    """POST query to Overpass API, return parsed JSON. Retries on failure."""
    resp = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=OVERPASS_TIMEOUT + 30,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_overpass_buildings(osm_json: dict) -> gpd.GeoDataFrame:
    """
    Convert Overpass JSON (ways/relations) to GeoDataFrame of building polygons.

    Each row = one building. Geometry = polygon. No address/personal data stored.
    Fields kept: osm_id, building_type, levels, height_m.
    """
    nodes = {el["id"]: el for el in osm_json["elements"] if el["type"] == "node"}
    features = []

    for el in osm_json["elements"]:
        if el["type"] not in ("way", "relation"):
            continue
        if "tags" not in el or "building" not in el["tags"]:
            continue

        # Build polygon from node refs (way only; skip complex relations)
        if el["type"] == "way":
            refs = el.get("nodes", [])
            coords = []
            for nid in refs:
                n = nodes.get(nid)
                if n:
                    coords.append((n["lon"], n["lat"]))
            if len(coords) < 4:  # need at least a triangle + closing
                continue
            from shapely.geometry import Polygon
            try:
                geom = Polygon(coords)
                if not geom.is_valid:
                    geom = geom.buffer(0)
            except Exception:
                continue
        else:
            continue  # skip complex relations for now

        tags = el["tags"]
        # Parse height: use height tag, then building:levels × 3m estimate
        height_m = None
        if "height" in tags:
            try:
                height_m = float(tags["height"].replace("m", "").strip())
            except ValueError:
                pass
        if height_m is None and "building:levels" in tags:
            try:
                height_m = float(tags["building:levels"]) * 3.0
            except ValueError:
                pass

        features.append({
            "osm_id": el["id"],
            "building_type": tags.get("building", "yes"),
            "height_m": height_m,
            "geometry": geom,
        })

    if not features:
        logger.warning("No building polygons parsed from Overpass response.")
        return gpd.GeoDataFrame(columns=["osm_id", "building_type", "height_m", "geometry"],
                                crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    logger.info(f"Parsed {len(gdf)} building polygons from OSM.")
    return gdf


def _parse_overpass_transit_stops(osm_json: dict) -> gpd.GeoDataFrame:
    """Convert Overpass JSON (nodes) to GeoDataFrame of transit stop points."""
    from shapely.geometry import Point

    features = []
    for el in osm_json["elements"]:
        if el["type"] != "node":
            continue
        tags = el.get("tags", {})
        features.append({
            "osm_id": el["id"],
            "name": tags.get("name", ""),
            "stop_type": (
                tags.get("railway") or
                tags.get("public_transport") or
                tags.get("highway", "unknown")
            ),
            "network": tags.get("network", ""),
            "geometry": Point(el["lon"], el["lat"]),
        })

    if not features:
        logger.warning("No transit stops parsed from Overpass response.")
        return gpd.GeoDataFrame(columns=["osm_id", "name", "stop_type", "network", "geometry"],
                                crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    logger.info(f"Parsed {len(gdf)} transit stops from OSM.")
    return gdf


# ── Street network (osmnx) ──────────────────────────────────────────────────────

def fetch_street_network(bbox: dict, output_path: Path, network_type: str = "all") -> None:
    """
    Download Chicago street network via osmnx and save as GraphML.

    network_type="all" includes streets, bike paths, pedestrian routes.
    This graph is used for computing walk distances to transit stations.
    """
    logger.info(f"Fetching street network for bbox: {bbox}")
    ox.settings.log_console = False
    ox.settings.use_cache = True

    G = ox.graph_from_bbox(
        north=bbox["north"],
        south=bbox["south"],
        east=bbox["east"],
        west=bbox["west"],
        network_type=network_type,
        simplify=True,
    )
    ox.save_graphml(G, filepath=str(output_path))
    n_nodes = len(G.nodes)
    n_edges = len(G.edges)
    logger.info(f"Street network saved: {n_nodes} nodes, {n_edges} edges → {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────────

def fetch_all(
    bbox: dict = CHICAGO_BBOX,
    output_dir: str | Path = "data/raw/",
    skip_street_network: bool = False,
) -> dict[str, Path]:
    """
    Fetch all OSM data layers for Chicago.

    Returns dict of {layer_name: output_path} for downstream processing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Buildings
    buildings_path = output_dir / "osm_buildings.geojson"
    if buildings_path.exists():
        logger.info(f"Buildings already fetched: {buildings_path} — skipping.")
    else:
        logger.info("Fetching OSM building footprints (may take 2-5 min)...")
        osm_json = _overpass_fetch(_buildings_query(bbox))
        buildings_gdf = _parse_overpass_buildings(osm_json)
        buildings_gdf.to_file(buildings_path, driver="GeoJSON")
        logger.info(f"Saved {len(buildings_gdf)} buildings → {buildings_path}")
    paths["buildings"] = buildings_path

    # 2. Transit stops
    stops_path = output_dir / "osm_transit_stops.geojson"
    if stops_path.exists():
        logger.info(f"Transit stops already fetched: {stops_path} — skipping.")
    else:
        logger.info("Fetching OSM transit stops...")
        osm_json = _overpass_fetch(_transit_stops_query(bbox))
        stops_gdf = _parse_overpass_transit_stops(osm_json)
        stops_gdf.to_file(stops_path, driver="GeoJSON")
        logger.info(f"Saved {len(stops_gdf)} transit stops → {stops_path}")
    paths["transit_stops"] = stops_path

    # 3. Street network
    streets_path = output_dir / "osm_streets.graphml"
    if skip_street_network:
        logger.info("Skipping street network (--skip-street-network flag set).")
    elif streets_path.exists():
        logger.info(f"Street network already fetched: {streets_path} — skipping.")
    else:
        fetch_street_network(bbox, streets_path)
    paths["streets"] = streets_path

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Chicago OSM data layers")
    parser.add_argument("--output", default="data/raw/", help="Output directory")
    parser.add_argument("--skip-street-network", action="store_true",
                        help="Skip large street network download (speeds up testing)")
    args = parser.parse_args()

    paths = fetch_all(
        bbox=CHICAGO_BBOX,
        output_dir=args.output,
        skip_street_network=args.skip_street_network,
    )
    for layer, path in paths.items():
        print(f"  {layer:20s} → {path}")
