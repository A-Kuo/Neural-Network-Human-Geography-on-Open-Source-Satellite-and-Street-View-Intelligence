"""Data fetchers for external data sources."""

from .base import BaseFetcher

# Optional imports for specific data sources
# Import these explicitly if needed to avoid missing optional dependencies
try:
    from .census import CensusFetcher
except ImportError:
    CensusFetcher = None

try:
    from .gtfs import GTFSFetcher
except ImportError:
    GTFSFetcher = None

try:
    from .osm import OSMFetcher
except ImportError:
    OSMFetcher = None

try:
    from .streetview import StreetViewFetcher
except ImportError:
    StreetViewFetcher = None

__all__ = [
    "BaseFetcher",
    "CensusFetcher",
    "GTFSFetcher",
    "OSMFetcher",
    "StreetViewFetcher",
]
