"""Base classes for data pipeline components."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ..config import Config


class DataFetcher(ABC):
    """Abstract base class for data fetchers.

    All fetchers should inherit from this and implement the fetch method.
    Fetchers are responsible for downloading raw data from external sources
    (Census API, OpenStreetMap, GTFS, Google Street View, etc.).
    """

    def __init__(self, config: Config):
        """Initialize fetcher.

        Args:
            config: Configuration instance with all settings
        """
        self.config = config
        self.logger = logger.bind(component=self.__class__.__name__)

    @abstractmethod
    def fetch(self) -> dict[str, Path]:
        """Fetch raw data from external source.

        Returns:
            Dictionary mapping output names to file paths
            Example: {'census_data': Path('data/raw/census/acs_2022.csv')}

        Raises:
            DataFetchError: If fetching fails
        """
        pass

    def _validate_output_paths(self, paths: dict[str, Path]) -> None:
        """Validate that all output files exist.

        Args:
            paths: Dictionary of output name → path

        Raises:
            FileNotFoundError: If any path doesn't exist
        """
        for name, path in paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} output not found: {path}")
            self.logger.info(f"  {name} → {path}")


class DataProcessor(ABC):
    """Abstract base class for data processors.

    Processors transform raw data into processed features:
    - Clean: handle missing values, format conversions
    - Aggregate: join multiple data sources
    - Validate: check privacy, fairness, quality constraints
    """

    def __init__(self, config: Config):
        """Initialize processor.

        Args:
            config: Configuration instance with all settings
        """
        self.config = config
        self.logger = logger.bind(component=self.__class__.__name__)

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process raw data into cleaned/aggregated form.

        Args:
            input_data: Raw data (format depends on processor)

        Returns:
            Processed data (format depends on processor)

        Raises:
            DataProcessingError: If processing fails
        """
        pass


class PipelineOutput:
    """Container for full pipeline execution results.

    Attributes:
        dataset: Final processed DataFrame/Dataset
        metadata: Dictionary with summary statistics and audit info
        audit_logs: List of audit/validation check results
        paths: Dictionary mapping output names to file paths
    """

    def __init__(
        self,
        dataset: Any,
        metadata: dict,
        audit_logs: list,
        paths: Optional[dict] = None,
    ):
        """Initialize pipeline output.

        Args:
            dataset: Final processed dataset
            metadata: Summary statistics and metadata
            audit_logs: List of audit results
            paths: Output file paths
        """
        self.dataset = dataset
        self.metadata = metadata
        self.audit_logs = audit_logs
        self.paths = paths or {}

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PipelineOutput("
            f"dataset_shape={getattr(self.dataset, 'shape', 'N/A')}, "
            f"metadata_keys={list(self.metadata.keys())}, "
            f"audit_checks={len(self.audit_logs)}, "
            f"files={len(self.paths)}"
            ")"
        )
