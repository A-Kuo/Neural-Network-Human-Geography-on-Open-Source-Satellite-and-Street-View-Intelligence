"""Configuration management for gnn_geography package."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .exceptions import ConfigurationError


@dataclass
class RegionConfig:
    """Geographic region configuration."""

    name: str
    state_fips: str
    county_fips: str
    bbox: dict
    loop_lat: float
    loop_lon: float
    utm_crs: str = "EPSG:32616"
    wgs84_crs: str = "EPSG:4326"


@dataclass
class PathsConfig:
    """Data directories configuration."""

    raw: str
    processed: str
    audit: str
    figures: str
    models: str
    logs: str
    osm_buildings: str = ""
    osm_streets: str = ""
    osm_transit_stops: str = ""
    gtfs_dir: str = ""
    gtfs_all_stops: str = ""
    census_dir: str = ""
    streetview_dir: str = ""
    image_features_dir: str = ""
    tract_centroids: str = ""
    tract_building_stats: str = ""
    tract_transit_features: str = ""
    tract_image_features: str = ""
    final_dataset: str = ""
    streetview_coverage_audit: str = ""
    streetview_image_log: str = ""
    feature_extraction_log: str = ""
    osm_coverage_audit: str = ""
    transit_coverage_audit: str = ""
    build_transform_log: str = ""

    def make_dirs(self) -> None:
        """Create all configured directories."""
        for attr in ["raw", "processed", "audit", "figures", "models", "logs"]:
            path = Path(getattr(self, attr))
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class CensusConfig:
    """Census/ACS API configuration."""

    acs_year: int
    acs_vintage: str
    api_base_url: str
    tiger_base_url: str
    variables: dict = field(default_factory=dict)


@dataclass
class StreetViewConfig:
    """Google Street View API configuration."""

    api_base_url: str
    metadata_url: str
    headings: list = field(default_factory=lambda: [0, 90, 180, 270])
    pitch: int = 0
    fov: int = 90
    image_size: str = "640x640"
    requests_per_second: int = 1
    images_per_tract_target: int = 30
    min_images_for_coverage: int = 10
    location_radius_m: int = 50


@dataclass
class OSMConfig:
    """OpenStreetMap configuration."""

    overpass_url: str
    overpass_timeout_s: int
    network_type: str = "all"
    simplify_graph: bool = True


@dataclass
class GTFSConfig:
    """GTFS/Transit configuration."""

    sources: dict = field(default_factory=dict)
    walk_speed_kmh: float = 5.0
    max_travel_time_min: int = 120
    loop_stop_radius_m: int = 600


@dataclass
class TransitConfig:
    """Transit accessibility configuration."""

    radius_500m: int
    radius_1km: int
    avg_hop_distance_km: float
    transit_score_weights: dict = field(default_factory=dict)


@dataclass
class FeaturesConfig:
    """Feature extraction configuration."""

    resnet_model: str
    resnet_feature_dim: int
    image_size: list = field(default_factory=lambda: [224, 224])
    imagenet_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    batch_size: int = 32
    min_images_per_tract: int = 10


@dataclass
class PrivacyConfig:
    """Privacy and ethics configuration."""

    min_tract_population: int
    min_samples_for_aggregation: int
    pii_column_patterns: list = field(default_factory=list)


@dataclass
class GNNConfig:
    """Graph neural network configuration."""

    hidden_dim: int
    depths: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    num_epochs: int = 200
    patience: int = 20
    dropout: float = 0.3
    knn_k: int = 4
    spatial_adjacency: bool = True
    train_years: list = field(default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023])
    test_year: int = 2024
    val_fraction: float = 0.2
    random_seed: int = 42


@dataclass
class SyntheticConfig:
    """Synthetic task configuration."""

    n_nodes: int
    graph_triangles: int
    num_hops: int
    income_scale: float
    noise_std: float
    num_node_features: int
    hidden_channels: int
    num_epochs: int
    num_trials: int
    seed: int = 42


@dataclass
class ReproducibilityConfig:
    """Reproducibility configuration."""

    global_seed: int
    numpy_seed: int
    torch_seed: int
    deterministic_cudnn: bool


@dataclass
class Config:
    """Main configuration class."""

    region: RegionConfig
    paths: PathsConfig
    census: CensusConfig
    streetview: StreetViewConfig
    osm: OSMConfig
    gtfs: GTFSConfig
    transit: TransitConfig
    features: FeaturesConfig
    privacy: PrivacyConfig
    gnn: GNNConfig
    synthetic: SyntheticConfig
    reproducibility: ReproducibilityConfig

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml file

        Returns:
            Config instance with all settings

        Raises:
            ConfigurationError: If file not found or invalid YAML
            FileNotFoundError: If config file does not exist
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}") from e

        if not isinstance(data, dict):
            raise ConfigurationError("Config file must contain a YAML mapping")

        return cls(
            region=RegionConfig(**data["region"]),
            paths=PathsConfig(**data["paths"]),
            census=CensusConfig(**data["census"]),
            streetview=StreetViewConfig(**data["streetview"]),
            osm=OSMConfig(**data["osm"]),
            gtfs=GTFSConfig(**data["gtfs"]),
            transit=TransitConfig(**data["transit"]),
            features=FeaturesConfig(**data["features"]),
            privacy=PrivacyConfig(**data["privacy"]),
            gnn=GNNConfig(**data["gnn"]),
            synthetic=SyntheticConfig(**data["synthetic"]),
            reproducibility=ReproducibilityConfig(**data["reproducibility"]),
        )

    @classmethod
    def from_env(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration, respecting environment overrides.

        Looks for config file in order:
        1. Environment variable GNNGEOGRAPHY_CONFIG
        2. Provided config_path parameter
        3. Current directory: config.yaml
        4. Project root: config.yaml

        Args:
            config_path: Optional explicit path to config file

        Returns:
            Config instance

        Raises:
            ConfigurationError: If no valid config found
        """
        # Try environment variable first
        if "GNNGEOGRAPHY_CONFIG" in os.environ:
            config_path = os.environ["GNNGEOGRAPHY_CONFIG"]
        elif config_path is None:
            # Look in standard locations
            candidates = [
                Path("config.yaml"),
                Path(__file__).parent.parent.parent / "config.yaml",
            ]
            config_path = None
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break

            if config_path is None:
                raise ConfigurationError(
                    "No config.yaml found. Set GNNGEOGRAPHY_CONFIG env var "
                    "or provide explicit path."
                )

        return cls.from_yaml(config_path)

    def set_up_directories(self) -> None:
        """Create all configured directories."""
        self.paths.make_dirs()


# Global config instance (lazy-loaded)
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config.from_env(config_path)
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance (for testing).

    Args:
        config: Config instance to use globally
    """
    global _config
    _config = config
