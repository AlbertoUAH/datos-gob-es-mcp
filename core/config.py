"""Centralized configuration from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_float(key: str, default: float) -> float:
    """Get float value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass
    return default


def _get_int(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def _get_str(key: str, default: str) -> str:
    """Get string value from environment variable."""
    return os.getenv(key, default)


def _get_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        return value.lower() in ("true", "1", "yes", "on")
    return default


# =============================================================================
# Base URLs
# =============================================================================
DATOS_GOB_BASE_URL = _get_str("DATOS_GOB_BASE_URL", "https://datos.gob.es/apidata/")
INE_BASE_URL = _get_str("INE_BASE_URL", "https://servicios.ine.es/wstempus/js/")
BOE_BASE_URL = _get_str("BOE_BASE_URL", "https://www.boe.es/datosabiertos/api/")
AEMET_BASE_URL = _get_str("AEMET_BASE_URL", "https://opendata.aemet.es/opendata/api/")

# =============================================================================
# HTTP Timeouts (seconds)
# =============================================================================
HTTP_DEFAULT_TIMEOUT = _get_float("HTTP_DEFAULT_TIMEOUT", 30.0)
PREVIEW_TIMEOUT = _get_float("PREVIEW_TIMEOUT", 10.0)
DOWNLOAD_TIMEOUT = _get_float("DOWNLOAD_TIMEOUT", 120.0)
RATE_LIMIT_TIMEOUT = _get_float("RATE_LIMIT_TIMEOUT", 30.0)

# =============================================================================
# Pagination and Results
# =============================================================================
DEFAULT_PAGE_SIZE = _get_int("DEFAULT_PAGE_SIZE", 50)
MAX_SEARCH_RESULTS = _get_int("MAX_SEARCH_RESULTS", 50)
DEFAULT_PREVIEW_ROWS = _get_int("DEFAULT_PREVIEW_ROWS", 10)
MAX_PREVIEW_ROWS = _get_int("MAX_PREVIEW_ROWS", 50)

# =============================================================================
# Cache Configuration
# =============================================================================
CACHE_TTL_HOURS = _get_int("CACHE_TTL_HOURS", 24)
CACHE_DIR = Path(os.path.expanduser(_get_str("CACHE_DIR", "~/.cache/datos-gob-es")))

# =============================================================================
# Download Limits
# =============================================================================
PREVIEW_MAX_BYTES = _get_int("PREVIEW_MAX_BYTES", 100 * 1024)  # 100KB
DOWNLOAD_MAX_BYTES = _get_int("DOWNLOAD_MAX_BYTES", 50 * 1024 * 1024)  # 50MB
MAX_DOWNLOAD_MB = _get_int("MAX_DOWNLOAD_MB", 50)

# =============================================================================
# Semantic Search
# =============================================================================
SEMANTIC_MODEL_NAME = _get_str("SEMANTIC_MODEL_NAME", "intfloat/multilingual-e5-small")
SEMANTIC_MIN_SCORE = _get_float("SEMANTIC_MIN_SCORE", 0.5)
SEMANTIC_TOP_K = _get_int("SEMANTIC_TOP_K", 20)
MAX_SEMANTIC_CANDIDATES = _get_int("MAX_SEMANTIC_CANDIDATES", 500)

# =============================================================================
# Processing Limits
# =============================================================================
DESCRIPTION_MAX_LENGTH = _get_int("DESCRIPTION_MAX_LENGTH", 200)
MAX_KEYWORDS = _get_int("MAX_KEYWORDS", 5)
PARALLEL_PAGES = _get_int("PARALLEL_PAGES", 5)
MAX_STATS_ROWS = _get_int("MAX_STATS_ROWS", 1000)

# =============================================================================
# INE Specific
# =============================================================================
INE_MAX_TABLES = _get_int("INE_MAX_TABLES", 100)
INE_MAX_DATA_RECORDS = _get_int("INE_MAX_DATA_RECORDS", 500)
INE_DEFAULT_NLAST = _get_int("INE_DEFAULT_NLAST", 10)

# =============================================================================
# BOE Specific
# =============================================================================
BOE_DEFAULT_SEARCH_DAYS = _get_int("BOE_DEFAULT_SEARCH_DAYS", 30)
BOE_MAX_SEARCH_DAYS = _get_int("BOE_MAX_SEARCH_DAYS", 90)
BOE_BATCH_SIZE = _get_int("BOE_BATCH_SIZE", 5)
BOE_MAX_RESULTS = _get_int("BOE_MAX_RESULTS", 50)

# =============================================================================
# AEMET Specific
# =============================================================================
AEMET_MAX_FORECAST_DAYS = _get_int("AEMET_MAX_FORECAST_DAYS", 7)
AEMET_MAX_OBSERVATIONS = _get_int("AEMET_MAX_OBSERVATIONS", 50)
AEMET_MAX_MUNICIPALITIES = _get_int("AEMET_MAX_MUNICIPALITIES", 500)

# =============================================================================
# HTTP Connection Pool (Performance)
# =============================================================================
HTTP_POOL_MAX_KEEPALIVE = _get_int("HTTP_POOL_MAX_KEEPALIVE", 10)
HTTP_POOL_MAX_CONNECTIONS = _get_int("HTTP_POOL_MAX_CONNECTIONS", 20)
HTTP2_ENABLED = _get_bool("HTTP2_ENABLED", True)

# =============================================================================
# Embeddings Performance
# =============================================================================
EMBEDDINGS_BATCH_SIZE = _get_int("EMBEDDINGS_BATCH_SIZE", 32)
PRELOAD_EMBEDDINGS_MODEL = _get_bool("PRELOAD_EMBEDDINGS_MODEL", False)
