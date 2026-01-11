"""Core module for logging, rate limiting, HTTP client utilities, and configuration."""

__version__ = "1.0.0"

from .client import BaseAPIClient
from .config import (
    AEMET_BASE_URL,
    # AEMET specific
    AEMET_MAX_FORECAST_DAYS,
    AEMET_MAX_MUNICIPALITIES,
    AEMET_MAX_OBSERVATIONS,
    BOE_BASE_URL,
    BOE_BATCH_SIZE,
    # BOE specific
    BOE_DEFAULT_SEARCH_DAYS,
    BOE_MAX_RESULTS,
    BOE_MAX_SEARCH_DAYS,
    CACHE_DIR,
    # Cache
    CACHE_TTL_HOURS,
    # Base URLs
    DATOS_GOB_BASE_URL,
    # Pagination
    DEFAULT_PAGE_SIZE,
    DEFAULT_PREVIEW_ROWS,
    # Processing limits
    DESCRIPTION_MAX_LENGTH,
    DOWNLOAD_MAX_BYTES,
    DOWNLOAD_TIMEOUT,
    # Embeddings Performance
    EMBEDDINGS_BATCH_SIZE,
    HTTP2_ENABLED,
    # Timeouts
    HTTP_DEFAULT_TIMEOUT,
    HTTP_POOL_MAX_CONNECTIONS,
    # HTTP Connection Pool
    HTTP_POOL_MAX_KEEPALIVE,
    INE_BASE_URL,
    INE_DEFAULT_NLAST,
    INE_MAX_DATA_RECORDS,
    # INE specific
    INE_MAX_TABLES,
    MAX_DOWNLOAD_MB,
    MAX_KEYWORDS,
    MAX_PREVIEW_ROWS,
    MAX_SEARCH_RESULTS,
    MAX_SEMANTIC_CANDIDATES,
    MAX_STATS_ROWS,
    PARALLEL_PAGES,
    PRELOAD_EMBEDDINGS_MODEL,
    # Download limits
    PREVIEW_MAX_BYTES,
    PREVIEW_TIMEOUT,
    RATE_LIMIT_TIMEOUT,
    SEMANTIC_MIN_SCORE,
    # Semantic search
    SEMANTIC_MODEL_NAME,
    SEMANTIC_TOP_K,
)
from .errors import handle_api_error
from .exceptions import (
    AEMETClientError,
    APIClientError,
    BOEClientError,
    DatosGobClientError,
    INEClientError,
)
from .http import HTTPClient
from .logging import get_logger, setup_logging
from .ratelimit import RateLimiter, RateLimitExceededError
from .utils import FORMAT_MAPPINGS, extract_dict_value, extract_uri_suffix, normalize_format

__all__ = [
    # Version
    "__version__",
    # Logging
    "setup_logging",
    "get_logger",
    # Rate limiting
    "RateLimiter",
    "RateLimitExceededError",
    # HTTP
    "HTTPClient",
    # Exceptions
    "APIClientError",
    "DatosGobClientError",
    "INEClientError",
    "BOEClientError",
    "AEMETClientError",
    # Error handling
    "handle_api_error",
    # Utilities
    "extract_uri_suffix",
    "extract_dict_value",
    "normalize_format",
    "FORMAT_MAPPINGS",
    # Base client
    "BaseAPIClient",
    # Configuration - Base URLs
    "DATOS_GOB_BASE_URL",
    "INE_BASE_URL",
    "BOE_BASE_URL",
    "AEMET_BASE_URL",
    # Configuration - Timeouts
    "HTTP_DEFAULT_TIMEOUT",
    "PREVIEW_TIMEOUT",
    "DOWNLOAD_TIMEOUT",
    "RATE_LIMIT_TIMEOUT",
    # Configuration - Pagination
    "DEFAULT_PAGE_SIZE",
    "MAX_SEARCH_RESULTS",
    "DEFAULT_PREVIEW_ROWS",
    "MAX_PREVIEW_ROWS",
    # Configuration - Cache
    "CACHE_TTL_HOURS",
    "CACHE_DIR",
    # Configuration - Download limits
    "PREVIEW_MAX_BYTES",
    "DOWNLOAD_MAX_BYTES",
    "MAX_DOWNLOAD_MB",
    # Configuration - Semantic search
    "SEMANTIC_MODEL_NAME",
    "SEMANTIC_MIN_SCORE",
    "SEMANTIC_TOP_K",
    "MAX_SEMANTIC_CANDIDATES",
    # Configuration - Processing limits
    "DESCRIPTION_MAX_LENGTH",
    "MAX_KEYWORDS",
    "PARALLEL_PAGES",
    "MAX_STATS_ROWS",
    # Configuration - INE specific
    "INE_MAX_TABLES",
    "INE_MAX_DATA_RECORDS",
    "INE_DEFAULT_NLAST",
    # Configuration - BOE specific
    "BOE_DEFAULT_SEARCH_DAYS",
    "BOE_MAX_SEARCH_DAYS",
    "BOE_BATCH_SIZE",
    "BOE_MAX_RESULTS",
    # Configuration - AEMET specific
    "AEMET_MAX_FORECAST_DAYS",
    "AEMET_MAX_OBSERVATIONS",
    "AEMET_MAX_MUNICIPALITIES",
    # Configuration - HTTP Connection Pool
    "HTTP_POOL_MAX_KEEPALIVE",
    "HTTP_POOL_MAX_CONNECTIONS",
    "HTTP2_ENABLED",
    # Configuration - Embeddings Performance
    "EMBEDDINGS_BATCH_SIZE",
    "PRELOAD_EMBEDDINGS_MODEL",
]
