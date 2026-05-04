"""GNN Geography: Graph Neural Networks for Human Geographic Analysis.

A research and production package for analyzing geographic patterns using
graph neural networks applied to Census, OpenStreetMap, transit, and
Street View data for Cook County, Chicago.
"""

from .__version__ import __author__, __description__, __version__
from .config import Config, get_config, set_config
from .exceptions import (
    ConfigurationError,
    DataFetchError,
    DataProcessingError,
    GNNGeographyError,
    PrivacyViolationError,
    RateLimitError,
    ValidationError,
)
from .pipeline import DataFetcher, DataProcessor, PipelineOutput

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "Config",
    "get_config",
    "set_config",
    "GNNGeographyError",
    "PrivacyViolationError",
    "ValidationError",
    "ConfigurationError",
    "DataFetchError",
    "DataProcessingError",
    "RateLimitError",
    "DataFetcher",
    "DataProcessor",
    "PipelineOutput",
]
