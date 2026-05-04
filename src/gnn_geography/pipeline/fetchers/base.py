"""Base class for all data fetchers."""

from abc import abstractmethod
from pathlib import Path

from loguru import logger

from ...config import Config
from ...pipeline.base import DataFetcher


class BaseFetcher(DataFetcher):
    """Base class for all data fetchers with common utilities."""

    def __init__(self, config: Config, source_name: str):
        """Initialize fetcher.

        Args:
            config: Configuration instance
            source_name: Name of data source (e.g., 'census', 'osm')
        """
        super().__init__(config)
        self.source_name = source_name
        self.logger = logger.bind(source=source_name)

    @abstractmethod
    def fetch(self) -> dict[str, Path]:
        """Fetch data from external source.

        Implemented by subclasses for each data source.
        """
        pass

    def _ensure_output_dir(self, output_dir: Path | str) -> Path:
        """Ensure output directory exists.

        Args:
            output_dir: Output directory path

        Returns:
            Path object for output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
