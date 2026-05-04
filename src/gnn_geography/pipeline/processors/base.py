"""Base class for all data processors."""

from abc import abstractmethod

from loguru import logger

from ...config import Config
from ...pipeline.base import DataProcessor


class BaseProcessor(DataProcessor):
    """Base class for all data processors with common utilities."""

    def __init__(self, config: Config, processor_name: str):
        """Initialize processor.

        Args:
            config: Configuration instance
            processor_name: Name of processor (e.g., 'osm_cleaner', 'transit_builder')
        """
        super().__init__(config)
        self.processor_name = processor_name
        self.logger = logger.bind(processor=processor_name)

    @abstractmethod
    def process(self, input_data):
        """Process input data.

        Implemented by subclasses for each processing step.
        """
        pass
