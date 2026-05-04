"""OpenStreetMap data fetcher."""

from pathlib import Path

import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import Point, Polygon
from tenacity import retry, stop_after_attempt, wait_exponential

from ...config import Config
from .base import BaseFetcher


def _bbox_str(bbox: dict) -> str:
    """Format bbox for Overpass QL: (south, west, north, east)."""
    return f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}"


def _buildings_query(bbox: dict, timeout: int = 120) -> str:
    """Overpass QL query for building footprints."""
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:{timeout}];
(
  way["building"]({bb});
  relation["building"]({bb});
);
out body;
>;
out skel qt;
"""


def _transit_stops_query(bbox: dict, timeout: int = 120) -> str:
    """Overpass QL query for CTA + Metra transit stops."""
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:{timeout}];
(
  node["railway"="station"]({bb});
  node["railway"="subway_entrance"]({bb});
  node["public_transport"="station"]({bb});
  node["public_transport"="stop_position"]({bb});
  node["highway"="bus_stop"]({bb});
);
out body;
"""


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
def _overpass_fetch(query: str, overpass_url: str = "https://overpass-api.de/api/interpreter",
                    timeout: int = 120) -> dict:
    """POST query to Overpass API, return parsed JSON."""
    resp = requests.post(
        overpass_url,
        data={"data": query},
        timeout=timeout + 30,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_overpass_buildings(osm_json: dict) -> gpd.GeoDataFrame:
    """Convert Overpass JSON (ways/relations) to GeoDataFrame of building polygons."""
    nodes = {el["id"]: el for el in osm_json["elements"] if el["type"] == "node"}
    features = []

    for el in osm_json["elements"]:
        if el["type"] not in ("way", "relation"):
            continue
        if "tags" not in el or "building" not in el["tags"]:
            continue

        # Build polygon from node refs
        if el["type"] == "way":
            refs = el.get("nodes", [])
            coords = []
            for nid in refs:
                n = nodes.get(nid)
                if n:
                    coords.append((n["lon"], n["lat"]))
            if len(coords) < 4:
                continue
            try:
                geom = Polygon(coords)
                if not geom.is_valid:
                    geom = geom.buffer(0)
            except Exception:
                continue
        else:
            continue  # skip complex relations

        tags = el["tags"]
        # Parse height
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

        features.append(
            {
                "osm_id": el["id"],
                "building_type": tags.get("building", "yes"),
                "height_m": height_m,
                "geometry": geom,
            }
        )

    if not features:
        return gpd.GeoDataFrame(
            columns=["osm_id", "building_type", "height_m", "geometry"], crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(features, crs="EPSG:4326")


def _parse_overpass_transit_stops(osm_json: dict) -> gpd.GeoDataFrame:
    """Convert Overpass JSON (nodes) to GeoDataFrame of transit stop points."""
    features = []
    for el in osm_json["elements"]:
        if el["type"] != "node":
            continue
        tags = el.get("tags", {})
        features.append(
            {
                "osm_id": el["id"],
                "name": tags.get("name", ""),
                "stop_type": (
                    tags.get("railway")
                    or tags.get("public_transport")
                    or tags.get("highway", "unknown")
                ),
                "network": tags.get("network", ""),
                "geometry": Point(el["lon"], el["lat"]),
            }
        )

    if not features:
        return gpd.GeoDataFrame(
            columns=["osm_id", "name", "stop_type", "network", "geometry"], crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(features, crs="EPSG:4326")


def fetch_street_network(
    bbox: dict, output_path: Path, network_type: str = "all"
) -> None:
    """Download Chicago street network via osmnx and save as GraphML."""
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


class OSMFetcher(BaseFetcher):
    """Fetcher for OpenStreetMap data (buildings, streets, transit stops)."""

    def __init__(self, config: Config):
        """Initialize OSM fetcher.

        Args:
            config: Configuration instance with OSM settings
        """
        super().__init__(config, "osm")

    def fetch(self) -> dict[str, Path]:
        """Fetch OSM building, transit, and street network data.

        Returns:
            Dictionary mapping layer names to file paths
        """
        osm_config = self.config.osm
        output_dir = Path(self.config.paths.raw)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Get bbox from config
        bbox = self.config.region.bbox

        # 1. Buildings
        buildings_path = output_dir / "osm_buildings.geojson"
        if buildings_path.exists():
            self.logger.info(f"Buildings already fetched: {buildings_path}")
        else:
            self.logger.info("Fetching OSM building footprints...")
            osm_json = _overpass_fetch(
                _buildings_query(bbox, timeout=osm_config.overpass_timeout_s),
                overpass_url=osm_config.overpass_url,
                timeout=osm_config.overpass_timeout_s,
            )
            buildings_gdf = _parse_overpass_buildings(osm_json)
            buildings_gdf.to_file(buildings_path, driver="GeoJSON")
            self.logger.info(f"Saved {len(buildings_gdf)} buildings → {buildings_path}")
        paths["buildings"] = buildings_path

        # 2. Transit stops
        stops_path = output_dir / "osm_transit_stops.geojson"
        if stops_path.exists():
            self.logger.info(f"Transit stops already fetched: {stops_path}")
        else:
            self.logger.info("Fetching OSM transit stops...")
            osm_json = _overpass_fetch(
                _transit_stops_query(bbox, timeout=osm_config.overpass_timeout_s),
                overpass_url=osm_config.overpass_url,
                timeout=osm_config.overpass_timeout_s,
            )
            stops_gdf = _parse_overpass_transit_stops(osm_json)
            stops_gdf.to_file(stops_path, driver="GeoJSON")
            self.logger.info(f"Saved {len(stops_gdf)} transit stops → {stops_path}")
        paths["transit_stops"] = stops_path

        # 3. Street network
        streets_path = output_dir / "osm_streets.graphml"
        if streets_path.exists():
            self.logger.info(f"Street network already fetched: {streets_path}")
        else:
            self.logger.info("Fetching street network (may take 2-5 min)...")
            fetch_street_network(bbox, streets_path, network_type=osm_config.network_type)
            self.logger.info(f"Street network saved → {streets_path}")
        paths["streets"] = streets_path

        return paths
