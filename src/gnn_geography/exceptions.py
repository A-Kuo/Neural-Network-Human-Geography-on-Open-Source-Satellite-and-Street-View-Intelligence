"""Custom exceptions for gnn_geography package."""


class GNNGeographyError(Exception):
    """Base exception for all gnn_geography errors."""

    pass


class PrivacyViolationError(GNNGeographyError):
    """Raised when a privacy rule is violated during data processing."""

    pass


class ValidationError(GNNGeographyError):
    """Raised when data validation fails."""

    pass


class ConfigurationError(GNNGeographyError):
    """Raised when configuration is invalid or incomplete."""

    pass


class DataFetchError(GNNGeographyError):
    """Raised when data fetching fails."""

    pass


class DataProcessingError(GNNGeographyError):
    """Raised when data processing fails."""

    pass


class RateLimitError(GNNGeographyError):
    """Raised when API rate limit is exceeded."""

    pass
