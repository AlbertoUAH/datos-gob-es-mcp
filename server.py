"""MCP server for datos.gob.es open data catalog API."""

import asyncio
import csv
import io
import json
import pickle
import re
import time
import unicodedata
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    """Remove accents and normalize text for API searches.

    The datos.gob.es API doesn't handle accented characters well,
    so we normalize them before searching.
    """
    # Normalize to NFD form (decomposed), remove combining characters (accents)
    normalized = unicodedata.normalize("NFD", text)
    without_accents = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return without_accents


import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Core utilities for logging, rate limiting, and configuration
from core import (
    CACHE_DIR,
    CACHE_TTL_HOURS,
    DATOS_GOB_BASE_URL,
    DEFAULT_PAGE_SIZE,
    DEFAULT_PREVIEW_ROWS,
    DESCRIPTION_MAX_LENGTH,
    DOWNLOAD_MAX_BYTES,
    DOWNLOAD_TIMEOUT,
    EMBEDDINGS_BATCH_SIZE,
    MAX_DOWNLOAD_MB,
    MAX_KEYWORDS,
    MAX_SEARCH_RESULTS,
    MAX_SEMANTIC_CANDIDATES,
    MAX_STATS_ROWS,
    PARALLEL_PAGES,
    PRELOAD_EMBEDDINGS_MODEL,
    PREVIEW_MAX_BYTES,
    PREVIEW_TIMEOUT,
    SEMANTIC_MIN_SCORE,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_TOP_K,
    BaseAPIClient,
    DatosGobClientError,
    get_logger,
    handle_api_error,
    setup_logging,
)

# Initialize structured logging
setup_logging()
logger = get_logger("datos_gob_es")

# Semantic search dependencies (lazy loaded on first use)
np = None
SentenceTransformer = None
EMBEDDINGS_AVAILABLE: bool | None = None  # None = not checked yet


def _load_embeddings_dependencies() -> bool:
    """Lazy load numpy and sentence_transformers on first use."""
    global np, SentenceTransformer, EMBEDDINGS_AVAILABLE

    if EMBEDDINGS_AVAILABLE is not None:
        return EMBEDDINGS_AVAILABLE

    try:
        import numpy as _np
        from sentence_transformers import SentenceTransformer as _ST

        np = _np
        SentenceTransformer = _ST
        EMBEDDINGS_AVAILABLE = True
        logger.info("embeddings_loaded", status="success")
    except ImportError as e:
        EMBEDDINGS_AVAILABLE = False
        logger.warning("embeddings_unavailable", error=str(e))

    return EMBEDDINGS_AVAILABLE


# =============================================================================
# KEYWORD EXTRACTION FOR SEMANTIC SEARCH
# =============================================================================

# Stopwords español/inglés básicas para filtrar de queries
STOPWORDS = {
    # Español
    "de",
    "la",
    "el",
    "en",
    "los",
    "las",
    "un",
    "una",
    "y",
    "o",
    "que",
    "del",
    "al",
    "es",
    "por",
    "con",
    "para",
    "sobre",
    "como",
    "su",
    "sus",
    "se",
    "no",
    "mas",
    "pero",
    "este",
    "esta",
    "esto",
    "estos",
    "estas",
    "mi",
    "tu",
    "nos",
    "les",
    "lo",
    "le",
    "me",
    "te",
    "si",
    "ya",
    "muy",
    "datos",
    "dataset",
    "datasets",
    "informacion",
    "buscar",
    "encontrar",
    "dame",
    "quiero",
    "necesito",
    "muestra",
    "mostrar",
    "ver",
    "todos",
    # Inglés
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "data",
    "find",
    "search",
    "show",
    "get",
    "all",
    "from",
    "about",
}


def _extract_keywords(query: str, max_keywords: int = MAX_KEYWORDS) -> list[str]:
    """Extrae keywords esenciales de una query eliminando stopwords.

    Args:
        query: Query natural del usuario.
        max_keywords: Máximo número de keywords a extraer.

    Returns:
        Lista de keywords filtradas y normalizadas.
    """
    # Normalizar: quitar acentos y convertir a minúsculas
    normalized = normalize_text(query.lower())
    # Extraer palabras (solo alfanuméricas)
    words = re.findall(r"\b[a-zA-Z0-9]+\b", normalized)
    # Filtrar stopwords y palabras muy cortas
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    # Eliminar duplicados manteniendo orden
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:max_keywords]


# =============================================================================
# MODELS
# =============================================================================


class PaginationParams(BaseModel):
    """Parameters for paginated API requests."""

    page: int = Field(default=0, ge=0, description="Page number (0-indexed)")
    page_size: int = Field(default=50, ge=1, le=50, description="Items per page (max 50)")
    sort: str | None = Field(
        default=None, description="Sort field(s), prefix with - for descending"
    )


class DatasetSummary(BaseModel):
    """Simplified dataset representation for responses."""

    uri: str
    id: str | None = None  # Short dataset ID extracted from URI
    title: str | list[str] | None = None
    description: str | list[str] | None = None
    publisher_name: str | None = None  # Human-readable publisher name
    theme: list[str] | str | None = None
    keywords: list[str] | None = None
    issued: str | None = None
    modified: str | None = None
    frequency: str | None = None  # Update frequency (accrualPeriodicity)
    language: list[str] | None = None  # Available languages
    spatial: str | None = None  # Geographic coverage
    license: str | None = None  # License information
    formats: list[str] | None = None  # Available formats (csv, json, etc.)
    access_url: str | None = None  # Main download URL
    distributions_count: int = 0
    preview: "DataPreview | None" = None  # Data preview (first 10 rows)

    @classmethod
    def from_api_item(cls, item: dict[str, Any], lang: str | None = "es") -> "DatasetSummary":
        """Create a DatasetSummary from an API response item.

        Args:
            item: The raw API response item.
            lang: Preferred language code ('es', 'en', 'ca', 'eu', 'gl').
                  Default is 'es' (Spanish). Use None to return all languages.
        """
        title = cls._extract_text(item.get("title"), lang)
        description = cls._extract_text(item.get("description"), lang)
        # Truncate long descriptions to reduce payload size
        if (
            description
            and isinstance(description, str)
            and len(description) > DESCRIPTION_MAX_LENGTH
        ):
            description = description[: DESCRIPTION_MAX_LENGTH - 3] + "..."

        distributions = item.get("distribution", [])
        if isinstance(distributions, dict):
            distributions = [distributions]

        theme = item.get("theme")
        if isinstance(theme, str):
            theme = [theme]
        elif isinstance(theme, list):
            theme = [t if isinstance(t, str) else str(t) for t in theme]

        keywords = cls._extract_keywords(item.get("keyword", []), lang)

        # Extract dataset ID from URI
        uri = item.get("_about", "")
        dataset_id = uri.split("/")[-1] if uri else None

        # Extract publisher name
        publisher = item.get("publisher")
        publisher_name = cls._extract_publisher_name(publisher)

        # Extract frequency (accrualPeriodicity)
        frequency = cls._extract_frequency(item.get("accrualPeriodicity"))

        # Extract languages
        language = cls._extract_languages(item.get("language"))

        # Extract spatial coverage
        spatial = cls._extract_spatial(item.get("spatial"))

        # Extract license
        license_val = cls._extract_license(item.get("license"))

        # Extract formats and access_url from distributions
        formats, access_url = cls._extract_distribution_info(distributions)

        return cls(
            uri=uri,
            id=dataset_id,
            title=title,
            description=description,
            publisher_name=publisher_name,
            theme=theme,
            keywords=keywords,
            issued=item.get("issued"),
            modified=item.get("modified"),
            frequency=frequency,
            language=language,
            spatial=spatial,
            license=license_val,
            formats=formats,
            access_url=access_url,
            distributions_count=len(distributions) if distributions else 0,
        )

    @staticmethod
    def _extract_publisher_name(publisher: Any) -> str | None:
        """Extract human-readable publisher name."""
        if publisher is None:
            return None
        if isinstance(publisher, str):
            # Try to extract from URL format
            if "/" in publisher:
                return publisher.split("/")[-1]
            return publisher
        if isinstance(publisher, dict):
            # Try common keys for publisher name
            for key in ["name", "title", "_value", "label"]:
                if key in publisher:
                    val = publisher[key]
                    if isinstance(val, dict):
                        return val.get("_value")
                    return val
            # Fallback to _about and extract last part
            about = publisher.get("_about", "")
            if about:
                return about.split("/")[-1]
        return None

    @staticmethod
    def _extract_frequency(value: Any) -> str | None:
        """Extract update frequency from accrualPeriodicity field."""
        if value is None:
            return None
        if isinstance(value, str):
            # Extract from URI format (e.g., http://purl.org/cld/freq/annual)
            if "/" in value:
                return value.split("/")[-1]
            return value
        if isinstance(value, dict):
            about = value.get("_about", "")
            if about and "/" in about:
                return about.split("/")[-1]
            return value.get("_value") or value.get("label")
        return None

    @staticmethod
    def _extract_languages(value: Any) -> list[str] | None:
        """Extract language codes from language field."""
        if value is None:
            return None
        languages = []
        if isinstance(value, str):
            # Extract from URI format
            if "/" in value:
                languages.append(value.split("/")[-1])
            else:
                languages.append(value)
        elif isinstance(value, list):
            for lang in value:
                if isinstance(lang, str):
                    if "/" in lang:
                        languages.append(lang.split("/")[-1])
                    else:
                        languages.append(lang)
                elif isinstance(lang, dict):
                    about = lang.get("_about", "")
                    if about and "/" in about:
                        languages.append(about.split("/")[-1])
        return languages if languages else None

    @staticmethod
    def _extract_spatial(value: Any) -> str | None:
        """Extract geographic coverage from spatial field."""
        if value is None:
            return None
        if isinstance(value, str):
            # Extract from URI format
            if "/" in value:
                return value.split("/")[-1]
            return value
        if isinstance(value, dict):
            about = value.get("_about", "")
            if about and "/" in about:
                return about.split("/")[-1]
            return value.get("_value") or value.get("label")
        if isinstance(value, list) and value:
            # Return first spatial coverage
            first = value[0]
            if isinstance(first, str):
                return first.split("/")[-1] if "/" in first else first
            if isinstance(first, dict):
                about = first.get("_about", "")
                if about:
                    return about.split("/")[-1]
        return None

    @staticmethod
    def _extract_license(value: Any) -> str | None:
        """Extract license information."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            # Try to get label or URI
            label = value.get("label") or value.get("_value") or value.get("title")
            if label:
                return label
            about = value.get("_about", "")
            if about:
                return about
        return None

    @staticmethod
    def _extract_distribution_info(distributions: list) -> tuple[list[str] | None, str | None]:
        """Extract formats and primary access URL from distributions."""
        if not distributions:
            return None, None

        formats = set()
        access_url = None

        for dist in distributions:
            if not isinstance(dist, dict):
                continue

            # Extract format
            fmt = dist.get("format")
            if isinstance(fmt, dict):
                fmt = fmt.get("value") or fmt.get("_value")
            elif isinstance(fmt, list) and fmt:
                f = fmt[0]
                fmt = f.get("value") if isinstance(f, dict) else f

            if fmt:
                # Normalize format names
                fmt_lower = fmt.lower()
                if "csv" in fmt_lower or "comma-separated" in fmt_lower:
                    formats.add("CSV")
                elif "json" in fmt_lower:
                    formats.add("JSON")
                elif "xml" in fmt_lower:
                    formats.add("XML")
                elif "xls" in fmt_lower or "excel" in fmt_lower or "spreadsheet" in fmt_lower:
                    formats.add("Excel")
                elif "pdf" in fmt_lower:
                    formats.add("PDF")
                elif "rdf" in fmt_lower:
                    formats.add("RDF")
                elif "html" in fmt_lower:
                    formats.add("HTML")
                elif "api" in fmt_lower:
                    formats.add("API")
                elif "zip" in fmt_lower:
                    formats.add("ZIP")
                elif "shp" in fmt_lower or "shapefile" in fmt_lower:
                    formats.add("Shapefile")
                elif "geojson" in fmt_lower:
                    formats.add("GeoJSON")
                elif "tsv" in fmt_lower or "tab-separated" in fmt_lower:
                    formats.add("TSV")
                elif "txt" in fmt_lower or "text/plain" in fmt_lower:
                    formats.add("TXT")
                else:
                    # Clean up the format name
                    clean_fmt = fmt.split("/")[-1] if "/" in fmt else fmt
                    formats.add(clean_fmt.upper() if len(clean_fmt) <= 5 else clean_fmt.title())

            # Get first access URL (prefer CSV or JSON for preview)
            if access_url is None:
                url = dist.get("accessURL")
                if url:
                    access_url = url

        return list(formats) if formats else None, access_url

    @staticmethod
    def _extract_keywords(value: Any, lang: str | None = None) -> list[str] | None:
        """Extract keywords handling multilingual format.

        Args:
            value: The keywords field from the API response.
            lang: Preferred language code. If None, returns all keywords.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            keywords = []
            for item in value:
                if isinstance(item, str):
                    keywords.append(item)
                elif isinstance(item, dict):
                    if lang and item.get("_lang") and item.get("_lang") != lang:
                        continue
                    keywords.append(item.get("_value", str(item)))
            return keywords if keywords else None
        return None

    @staticmethod
    def _extract_text(value: Any, lang: str | None = None) -> str | list[str] | None:
        """Extract text from multilingual field, optionally filtering by language.

        Args:
            value: The field value from the API response.
            lang: Preferred language code ('es', 'en', 'ca', 'eu', 'gl').
                  If None, returns all language versions.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if lang and value.get("_lang") and value.get("_lang") != lang:
                return None
            return value.get("_value", str(value))
        if isinstance(value, list):
            texts = []
            for item in value:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    if lang and item.get("_lang") and item.get("_lang") != lang:
                        continue
                    texts.append(item.get("_value", str(item)))
            return texts[0] if len(texts) == 1 else (texts if texts else None)
        return str(value)


class ColumnStats(BaseModel):
    """Statistics for a single column."""

    name: str
    inferred_type: (
        str  # "string", "integer", "float", "boolean", "date", "datetime", "null", "mixed"
    )
    null_count: int = 0
    unique_count: int | None = None  # None if too many to count
    sample_values: list[Any] | None = None  # Up to 5 sample non-null values
    min_value: Any | None = None  # For numeric/date columns
    max_value: Any | None = None  # For numeric/date columns


class DataPreview(BaseModel):
    """Preview of actual data content from a distribution."""

    columns: list[str]
    rows: list[list[Any]]
    total_rows: int | None = None
    total_columns: int | None = None
    file_size_bytes: int | None = None
    format: str
    truncated: bool = False
    error: str | None = None
    column_stats: list[ColumnStats] | None = None  # Statistics per column


class DistributionSummary(BaseModel):
    """Simplified distribution representation."""

    uri: str
    title: str | None = None
    access_url: str | None = None
    format: str | None = None
    media_type: str | None = None
    preview: DataPreview | None = None

    @classmethod
    def from_api_item(cls, item: dict[str, Any]) -> "DistributionSummary":
        """Create a DistributionSummary from an API response item."""
        title = item.get("title")
        if isinstance(title, dict):
            title = title.get("_value")
        elif isinstance(title, list) and title:
            title = title[0].get("_value") if isinstance(title[0], dict) else title[0]

        # Handle format field that can be string or dict
        format_val = item.get("format")
        if isinstance(format_val, dict):
            format_val = format_val.get("value") or format_val.get("_value")
        elif isinstance(format_val, list) and format_val:
            fmt = format_val[0]
            format_val = fmt.get("value") if isinstance(fmt, dict) else fmt

        # Handle mediaType field that can be string or dict
        media_type = item.get("mediaType")
        if isinstance(media_type, dict):
            media_type = media_type.get("value") or media_type.get("_value")

        return cls(
            uri=item.get("_about", ""),
            title=title,
            access_url=item.get("accessURL"),
            format=format_val,
            media_type=media_type,
        )


# =============================================================================
# SEMANTIC RANKER FOR RE-RANKING SEARCH RESULTS
# =============================================================================


class SemanticRanker:
    """Re-rankea resultados de búsqueda usando similaridad semántica.

    Usa intfloat/multilingual-e5-small para calcular embeddings y similaridad.
    No requiere índice pre-construido - calcula similaridad on-demand.

    El modelo E5 requiere:
    - "query: " prefix para queries de búsqueda
    - "passage: " prefix para documentos/textos
    """

    MODEL_NAME = SEMANTIC_MODEL_NAME

    def __init__(self):
        self.model = None

    def _load_model(self):
        """Carga lazy del modelo de embeddings."""
        if not _load_embeddings_dependencies():
            raise RuntimeError(
                "Semantic search requires sentence-transformers and numpy. "
                "Install with: pip install sentence-transformers numpy"
            )
        if self.model is None:
            self.model = SentenceTransformer(self.MODEL_NAME)

    def compute_similarity(self, query: str, text: str) -> float:
        """Calcula similaridad coseno entre una query y un texto.

        Args:
            query: Query de búsqueda.
            text: Texto a comparar.

        Returns:
            Score de similaridad (0-1).
        """
        self._load_model()

        # Encode con prefijos E5 (normalizados)
        query_embedding = self.model.encode(f"query: {query}", normalize_embeddings=True)
        text_embedding = self.model.encode(f"passage: {text}", normalize_embeddings=True)

        # Similaridad coseno (ya normalizados)
        similarity = float(np.dot(query_embedding, text_embedding))
        return max(0.0, similarity)

    def rank_results(
        self,
        query: str,
        datasets: list[dict[str, Any]],
        min_score: float = SEMANTIC_MIN_SCORE,
        top_k: int = SEMANTIC_TOP_K,
    ) -> list[dict[str, Any]]:
        """Re-rankea datasets por similaridad semántica con la query.

        Uses batch encoding for much better performance compared to one-by-one.

        Args:
            query: Query original del usuario.
            datasets: Lista de datasets a re-rankear.
            min_score: Score mínimo para incluir en resultados (default 0.5).
            top_k: Máximo número de resultados a devolver.

        Returns:
            Lista de datasets ordenados por semantic_score descendente.
        """
        if not datasets:
            return []

        self._load_model()

        # Build texts for batch encoding
        texts = []
        for ds in datasets:
            title = ds.get("title", "")
            description = ds.get("description", "")
            keywords = " ".join(ds.get("keywords", []) or [])
            texts.append(f"{title} {description} {keywords}".strip())

        # Filter out empty texts and keep track of indices
        valid_indices = [i for i, t in enumerate(texts) if t]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            return []

        # Batch encode - MUCH faster than loop
        query_embedding = self.model.encode(f"query: {query}", normalize_embeddings=True)
        text_embeddings = self.model.encode(
            [f"passage: {t}" for t in valid_texts],
            normalize_embeddings=True,
            batch_size=EMBEDDINGS_BATCH_SIZE,
            show_progress_bar=False,
        )

        # Calculate similarities using vectorized operation
        scores = np.dot(text_embeddings, query_embedding)

        # Filter and build results
        ranked = []
        for idx, (orig_idx, score) in enumerate(zip(valid_indices, scores)):
            if score >= min_score:
                ds = datasets[orig_idx]
                ds["semantic_score"] = round(float(score), 4)
                ranked.append(ds)

        # Sort by score descending
        ranked.sort(key=lambda x: x.get("semantic_score", 0), reverse=True)
        return ranked[:top_k]


# =============================================================================
# METADATA CACHE FOR STATIC DATA
# =============================================================================


class MetadataCache:
    """Cache for static metadata (publishers, themes, regions, provinces).

    Caches data that rarely changes to reduce API calls. Cache expires after TTL.
    """

    _CACHE_DIR = CACHE_DIR
    CACHE_FILE = _CACHE_DIR / "metadata.pkl"
    _CACHE_TTL = CACHE_TTL_HOURS * 60 * 60  # Convert hours to seconds

    def __init__(self):
        self.publishers: list[dict] | None = None
        self.themes: list[dict] | None = None
        self.provinces: list[dict] | None = None
        self.autonomous_regions: list[dict] | None = None
        self.public_sectors: list[dict] | None = None
        self.spatial_coverage: list[dict] | None = None
        self._cache_timestamp: float | None = None
        self._load_cache()

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid based on TTL."""
        if self._cache_timestamp is None:
            return False
        return (time.time() - self._cache_timestamp) < self._CACHE_TTL

    def _load_cache(self) -> bool:
        """Load metadata from cache file if it exists and is valid.

        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        if not self.CACHE_FILE.exists():
            return False

        try:
            with open(self.CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)

            self._cache_timestamp = cache_data.get("timestamp")
            if not self._is_cache_valid():
                return False

            self.publishers = cache_data.get("publishers")
            self.themes = cache_data.get("themes")
            self.provinces = cache_data.get("provinces")
            self.autonomous_regions = cache_data.get("autonomous_regions")
            self.public_sectors = cache_data.get("public_sectors")
            self.spatial_coverage = cache_data.get("spatial_coverage")
            logger.info(
                "metadata_cache_loaded",
                age_hours=round((time.time() - self._cache_timestamp) / 3600, 1),
            )
            return True
        except Exception as e:
            logger.warning("metadata_cache_load_failed", error=str(e))
            return False

    def _save_cache(self):
        """Save metadata to cache file."""
        self._ensure_cache_dir()
        cache_data = {
            "timestamp": self._cache_timestamp,
            "publishers": self.publishers,
            "themes": self.themes,
            "provinces": self.provinces,
            "autonomous_regions": self.autonomous_regions,
            "public_sectors": self.public_sectors,
            "spatial_coverage": self.spatial_coverage,
        }
        try:
            with open(self.CACHE_FILE, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info("metadata_cache_saved")
        except Exception as e:
            logger.warning("metadata_cache_save_failed", error=str(e))

    def clear(self):
        """Clear all cached data and delete cache file."""
        self.publishers = None
        self.themes = None
        self.provinces = None
        self.autonomous_regions = None
        self.public_sectors = None
        self.spatial_coverage = None
        self._cache_timestamp = None
        if self.CACHE_FILE.exists():
            self.CACHE_FILE.unlink()
        logger.info("metadata_cache_cleared")

    async def _fetch_all_paginated(
        self,
        client: "DatosGobClient",
        fetch_method: str,
        cache_attr: str,
    ) -> list[dict]:
        """Generic method to fetch all pages of a metadata type.

        Args:
            client: The API client instance.
            fetch_method: Name of the client method to call.
            cache_attr: Name of the cache attribute to store results.

        Returns:
            List of all fetched items.
        """
        # Check cache first
        cached = getattr(self, cache_attr, None)
        if cached and self._is_cache_valid():
            return cached

        # Fetch all pages
        all_items: list[dict] = []
        page = 0
        method = getattr(client, fetch_method)

        while True:
            pagination = PaginationParams(page=page, page_size=50)
            data = await method(pagination)
            items = data.get("result", {}).get("items", [])
            if not items:
                break
            all_items.extend(items)
            if len(items) < 50:
                break
            page += 1

        # Update cache
        setattr(self, cache_attr, all_items)
        self._cache_timestamp = time.time()
        self._save_cache()

        return all_items

    async def get_publishers(self, client: "DatosGobClient") -> list[dict]:
        """Get publishers, using cache if available."""
        return await self._fetch_all_paginated(client, "list_publishers", "publishers")

    async def get_themes(self, client: "DatosGobClient") -> list[dict]:
        """Get themes, using cache if available."""
        return await self._fetch_all_paginated(client, "list_themes", "themes")

    async def get_provinces(self, client: "DatosGobClient") -> list[dict]:
        """Get provinces, using cache if available."""
        return await self._fetch_all_paginated(client, "list_provinces", "provinces")

    async def get_autonomous_regions(self, client: "DatosGobClient") -> list[dict]:
        """Get autonomous regions, using cache if available."""
        return await self._fetch_all_paginated(
            client, "list_autonomous_regions", "autonomous_regions"
        )

    async def get_public_sectors(self, client: "DatosGobClient") -> list[dict]:
        """Get public sectors, using cache if available."""
        return await self._fetch_all_paginated(client, "list_public_sectors", "public_sectors")

    async def get_spatial_coverage(self, client: "DatosGobClient") -> list[dict]:
        """Get spatial coverage options, using cache if available."""
        return await self._fetch_all_paginated(client, "list_spatial_coverage", "spatial_coverage")

    async def preload_all(self, client: "DatosGobClient") -> None:
        """Load all metadata in parallel for better performance.

        This method fetches all metadata types concurrently, which is faster
        than fetching them sequentially when multiple types are needed.
        """
        if self._is_cache_valid() and all(
            [
                self.publishers,
                self.themes,
                self.provinces,
                self.autonomous_regions,
                self.public_sectors,
                self.spatial_coverage,
            ]
        ):
            logger.debug("metadata_preload_skipped", reason="cache_valid")
            return

        logger.info("metadata_preload_start")
        start_time = time.time()

        results = await asyncio.gather(
            self.get_publishers(client),
            self.get_themes(client),
            self.get_provinces(client),
            self.get_autonomous_regions(client),
            self.get_public_sectors(client),
            self.get_spatial_coverage(client),
            return_exceptions=True,
        )

        # Log any failures
        metadata_names = [
            "publishers",
            "themes",
            "provinces",
            "autonomous_regions",
            "public_sectors",
            "spatial_coverage",
        ]
        for name, result in zip(metadata_names, results):
            if isinstance(result, Exception):
                logger.warning("metadata_preload_partial_failure", metadata=name, error=str(result))

        duration_ms = (time.time() - start_time) * 1000
        logger.info("metadata_preload_complete", duration_ms=round(duration_ms, 2))


# Global semantic ranker instance
semantic_ranker = SemanticRanker()

# Global metadata cache instance
metadata_cache = MetadataCache()

# Optional: Pre-load embeddings model at startup for faster first search
if PRELOAD_EMBEDDINGS_MODEL:
    try:
        semantic_ranker._load_model()
        logger.info("embeddings_model_preloaded", model=SEMANTIC_MODEL_NAME)
    except Exception as e:
        logger.warning("embeddings_model_preload_failed", error=str(e))


# =============================================================================
# USAGE METRICS - Track tool and dataset usage (Proposal 19)
# =============================================================================


class UsageMetrics:
    """Track usage statistics for tools and datasets.

    Stores metrics in a local file for persistence across sessions.
    """

    METRICS_DIR = Path.home() / ".cache" / "datos-gob-es"
    METRICS_FILE = METRICS_DIR / "usage_metrics.json"

    def __init__(self):
        self.tool_calls: dict[str, int] = {}
        self.dataset_accesses: dict[str, int] = {}
        self.search_queries: list[dict[str, Any]] = []
        self.session_start: float = time.time()
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load metrics from disk if available."""
        try:
            if self.METRICS_FILE.exists():
                with open(self.METRICS_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                    self.tool_calls = data.get("tool_calls", {})
                    self.dataset_accesses = data.get("dataset_accesses", {})
                    # Keep only last 100 search queries
                    self.search_queries = data.get("search_queries", [])[-100:]
        except Exception as e:
            # Start fresh if load fails
            logger.warning("metrics_load_failed", error=str(e))

    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            self.METRICS_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "tool_calls": self.tool_calls,
                "dataset_accesses": self.dataset_accesses,
                "search_queries": self.search_queries[-100:],  # Keep last 100
                "last_updated": time.time(),
            }
            with open(self.METRICS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("metrics_save_failed", error=str(e))

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool invocation."""
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
        self._save_metrics()

    def record_dataset_access(self, dataset_id: str) -> None:
        """Record a dataset access."""
        self.dataset_accesses[dataset_id] = self.dataset_accesses.get(dataset_id, 0) + 1
        self._save_metrics()

    def record_search(self, query_params: dict[str, Any]) -> None:
        """Record a search query."""
        self.search_queries.append(
            {
                "timestamp": time.time(),
                "params": {k: v for k, v in query_params.items() if v is not None},
            }
        )
        self._save_metrics()

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        # Top 10 most used tools
        top_tools = sorted(self.tool_calls.items(), key=lambda x: x[1], reverse=True)[:10]

        # Top 10 most accessed datasets
        top_datasets = sorted(self.dataset_accesses.items(), key=lambda x: x[1], reverse=True)[:10]

        # Recent search patterns
        recent_searches = self.search_queries[-10:]

        return {
            "total_tool_calls": sum(self.tool_calls.values()),
            "unique_tools_used": len(self.tool_calls),
            "top_tools": [{"tool": t, "calls": c} for t, c in top_tools],
            "total_dataset_accesses": sum(self.dataset_accesses.values()),
            "unique_datasets_accessed": len(self.dataset_accesses),
            "top_datasets": [{"dataset_id": d, "accesses": c} for d, c in top_datasets],
            "total_searches": len(self.search_queries),
            "recent_searches": recent_searches,
            "session_duration_minutes": round((time.time() - self.session_start) / 60, 2),
        }

    def clear(self) -> None:
        """Clear all metrics."""
        self.tool_calls = {}
        self.dataset_accesses = {}
        self.search_queries = []
        self.session_start = time.time()
        self._save_metrics()


# Global usage metrics instance
usage_metrics = UsageMetrics()


# =============================================================================
# HTTP CLIENT
# =============================================================================


# DatosGobClientError is now imported from core.exceptions


class DatosGobClient(BaseAPIClient):
    """Async HTTP client for the datos.gob.es API.

    Uses BaseAPIClient for automatic logging and rate limiting.
    Overrides _request to add .json extension to endpoints.
    """

    BASE_URL = DATOS_GOB_BASE_URL
    API_NAME = "datos.gob.es"
    ERROR_CLASS = DatosGobClientError

    def _build_params(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        """Build query parameters from pagination settings."""
        params: dict[str, Any] = {}
        if pagination:
            params["_page"] = pagination.page
            params["_pageSize"] = pagination.page_size
            if pagination.sort:
                params["_sort"] = pagination.sort
        return params

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make an async HTTP request to the API with logging and rate limiting."""
        # Add .json extension if not present
        if not endpoint.endswith(".json"):
            endpoint = f"{endpoint}.json"

        try:
            return await self.http.get_json(endpoint, params=params)
        except Exception as e:
            # Re-raise as DatosGobClientError for backwards compatibility
            status_code = getattr(e, "status_code", None)
            raise DatosGobClientError(str(e), status_code=status_code) from e

    async def list_datasets(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("catalog/dataset", params)

    async def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return await self._request(f"catalog/dataset/{dataset_id}")

    async def search_datasets_by_title(
        self, title: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/title/{title}", params)

    async def get_datasets_by_publisher(
        self, publisher_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/publisher/{publisher_id}", params)

    async def get_datasets_by_theme(
        self, theme_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/theme/{theme_id}", params)

    async def get_datasets_by_format(
        self, format_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/format/{format_id}", params)

    async def get_datasets_by_keyword(
        self, keyword: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/keyword/{keyword}", params)

    async def get_datasets_by_spatial(
        self,
        spatial_word1: str,
        spatial_word2: str,
        pagination: PaginationParams | None = None,
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request(
            f"catalog/dataset/spatial/{spatial_word1}/{spatial_word2}", params
        )

    async def get_datasets_by_date_range(
        self,
        begin_date: str,
        end_date: str,
        pagination: PaginationParams | None = None,
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request(
            f"catalog/dataset/modified/begin/{begin_date}/end/{end_date}", params
        )

    async def list_distributions(
        self, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("catalog/distribution", params)

    async def get_distributions_by_dataset(
        self, dataset_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request(f"catalog/distribution/dataset/{dataset_id}", params)

    async def get_distributions_by_format(
        self, format_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request(f"catalog/distribution/format/{format_id}", params)

    async def list_publishers(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("catalog/publisher", params)

    async def list_spatial_coverage(
        self, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("catalog/spatial", params)

    async def list_themes(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("catalog/theme", params)

    async def list_public_sectors(
        self, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("nti/public-sector", params)

    async def get_public_sector(self, sector_id: str) -> dict[str, Any]:
        return await self._request(f"nti/public-sector/{sector_id}")

    async def list_provinces(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("nti/territory/Province", params)

    async def get_province(self, province_id: str) -> dict[str, Any]:
        return await self._request(f"nti/territory/Province/{province_id}")

    async def list_autonomous_regions(
        self, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("nti/territory/Autonomous-region", params)

    async def get_autonomous_region(self, region_id: str) -> dict[str, Any]:
        return await self._request(f"nti/territory/Autonomous-region/{region_id}")

    async def get_country_spain(self) -> dict[str, Any]:
        return await self._request("nti/territory/Country/España")


# =============================================================================
# MCP SERVER
# =============================================================================

# Metadata reference - use these IDs with the search() tool
METADATA_REFERENCE = """
## Metadata Reference for search() filters

### Themes (use with theme= parameter)
economia, hacienda, educacion, salud, medio-ambiente, transporte, turismo,
empleo, sector-publico, ciencia-tecnologia, cultura-ocio, urbanismo-infraestructuras, energia

### Main Publishers (use with publisher= parameter)
- EA0010587: INE (Instituto Nacional de Estadística)
- E05024401: Ministerio de Hacienda
- E05024301: Ministerio de Economía
- E00003901: AEMET (Agencia Estatal de Meteorología)
- L01280796: Ayuntamiento de Madrid
- L01080193: Ajuntament de Barcelona
- A08002970: Generalitat de Catalunya
- A01002820: Gobierno Vasco
- A13002908: Junta de Andalucía

### Province Codes (first 2 digits of municipality codes)
01-Álava, 02-Albacete, 03-Alicante, 04-Almería, 05-Ávila, 06-Badajoz, 07-Baleares,
08-Barcelona, 09-Burgos, 10-Cáceres, 11-Cádiz, 12-Castellón, 13-Ciudad Real,
14-Córdoba, 15-A Coruña, 16-Cuenca, 17-Girona, 18-Granada, 19-Guadalajara,
20-Guipúzcoa, 21-Huelva, 22-Huesca, 23-Jaén, 24-León, 25-Lleida, 26-La Rioja,
27-Lugo, 28-Madrid, 29-Málaga, 30-Murcia, 31-Navarra, 32-Ourense, 33-Asturias,
34-Palencia, 35-Las Palmas, 36-Pontevedra, 37-Salamanca, 38-Santa Cruz de Tenerife,
39-Cantabria, 40-Segovia, 41-Sevilla, 42-Soria, 43-Tarragona, 44-Teruel, 45-Toledo,
46-Valencia, 47-Valladolid, 48-Vizcaya, 49-Zamora, 50-Zaragoza, 51-Ceuta, 52-Melilla
"""

mcp = FastMCP(
    "datos-gob-es",
    instructions=METADATA_REFERENCE,
)

client = DatosGobClient()


def _format_response(
    data: dict[str, Any], summary_type: str | None = None, lang: str | None = "es"
) -> str:
    """Format API response for readable output.

    Args:
        data: The raw API response.
        summary_type: Type of summary to generate ('dataset', 'distribution', or None).
        lang: Preferred language code ('es', 'en', 'ca', 'eu', 'gl').
              Default is 'es' (Spanish). Use None to return all languages.
    """
    result = data.get("result", {})
    items = result.get("items", [])

    output = {
        "total_in_page": len(items),
        "page": result.get("page", 0),
        "items_per_page": result.get("itemsPerPage", 10),
    }

    if summary_type == "dataset":
        output["datasets"] = [
            DatasetSummary.from_api_item(item, lang).model_dump(exclude_none=True) for item in items
        ]
    elif summary_type == "distribution":
        output["distributions"] = [
            DistributionSummary.from_api_item(item).model_dump(exclude_none=True) for item in items
        ]
    else:
        output["items"] = items

    return json.dumps(output, ensure_ascii=False, indent=2)


def _handle_error(e: Exception) -> str:
    """Format error message."""
    return handle_api_error(e, context="datos_gob_operation", logger_name="datos_gob_es")


async def _fetch_all_pages(
    fetch_fn,
    max_results: int = MAX_SEARCH_RESULTS,
    sort: str | None = None,
    parallel_pages: int = PARALLEL_PAGES,
    **kwargs,
) -> list[dict[str, Any]]:
    """Fetch all pages from an API endpoint up to max_results.

    Uses parallel requests to speed up fetching when possible.

    Args:
        fetch_fn: The client method to call for fetching data.
        max_results: Maximum number of results to fetch.
        sort: Sort field for the query.
        parallel_pages: Number of pages to fetch in parallel (default 5).
        **kwargs: Additional arguments to pass to fetch_fn.

    Returns:
        List of all items fetched across multiple pages.
    """
    all_items: list[dict[str, Any]] = []
    page = 0

    while len(all_items) < max_results:
        # Create batch of page fetches
        tasks = []
        for i in range(parallel_pages):
            current_page = page + i
            pagination = PaginationParams(page=current_page, page_size=DEFAULT_PAGE_SIZE, sort=sort)
            tasks.append(fetch_fn(pagination=pagination, **kwargs))

        # Fetch pages in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results in order
        got_items = False
        last_page_partial = False
        for result in results:
            if isinstance(result, Exception):
                continue
            items = result.get("result", {}).get("items", [])
            if items:
                all_items.extend(items)
                got_items = True
                # Check if this page was partial (less than full page)
                if len(items) < DEFAULT_PAGE_SIZE:
                    last_page_partial = True

        # Stop if no items were returned or we hit a partial page
        if not got_items or last_page_partial:
            break

        page += parallel_pages

    return all_items[:max_results]


def _extract_text_for_match(value: Any) -> str:
    """Extract text from a field for matching purposes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("_value", str(value))
    if isinstance(value, list):
        texts = []
        for item in value:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(item.get("_value", str(item)))
        return " ".join(texts)
    return str(value)


def _matches_exact_word(text: str, search_term: str) -> bool:
    """Check if text contains the search term as a complete word."""
    # Use word boundary regex for exact word matching
    # \b matches word boundaries (start/end of word)
    pattern = rf"\b{re.escape(search_term)}\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _filter_items_by_exact_match(
    items: list[dict[str, Any]], search_term: str, fields: list[str] | None = None
) -> list[dict[str, Any]]:
    """Filter items to only include those with exact word matches."""
    if fields is None:
        fields = ["title", "description"]
    filtered = []
    for item in items:
        for field in fields:
            text = _extract_text_for_match(item.get(field))
            if _matches_exact_word(text, search_term):
                filtered.append(item)
                break  # Item matches, no need to check other fields
    return filtered


def _filter_datasets_locally(
    items: list[dict[str, Any]],
    publisher: str | None = None,
    theme: str | None = None,
    themes: list[str] | None = None,
    format_filter: str | None = None,
    keyword: str | None = None,
    title: str | None = None,
    license_filter: str | None = None,
    frequency_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Apply local filtering to dataset items after API query.

    Args:
        items: List of dataset items from API.
        publisher: Filter by publisher ID.
        theme: Filter by single theme (for backwards compatibility).
        themes: Filter by multiple themes (OR logic).
        format_filter: Filter by format.
        keyword: Filter by keyword.
        title: Filter by title.
        license_filter: Filter by license.
        frequency_filter: Filter by update frequency.
    """
    if not any(
        [publisher, theme, themes, format_filter, keyword, title, license_filter, frequency_filter]
    ):
        return items

    filtered = []
    for item in items:
        # Check publisher filter
        if publisher:
            item_publisher = item.get("publisher", {})
            if isinstance(item_publisher, dict):
                pub_id = item_publisher.get("_about", "")
            else:
                pub_id = str(item_publisher)
            if publisher.lower() not in pub_id.lower():
                continue

        # Check theme filter (single or multiple with OR logic)
        if theme or themes:
            # Combine single theme and multiple themes
            all_search_themes: list[str] = []
            if theme:
                all_search_themes.append(theme.lower())
            if themes:
                all_search_themes.extend(t.lower() for t in themes)

            item_themes = item.get("theme", [])
            if isinstance(item_themes, str):
                item_themes = [item_themes]

            # Match ANY of the search themes (OR logic)
            theme_match = any(
                any(search_theme in t.lower() for t in item_themes)
                for search_theme in all_search_themes
            )
            if not theme_match:
                continue

        # Check format filter (in distributions)
        if format_filter:
            distributions = item.get("distribution", [])
            if isinstance(distributions, dict):
                distributions = [distributions]
            format_match = False
            for dist in distributions:
                dist_format = dist.get("format", "")
                if isinstance(dist_format, dict):
                    dist_format = dist_format.get("value", "")
                if format_filter.lower() in dist_format.lower():
                    format_match = True
                    break
            if not format_match:
                continue

        # Check title filter (searches in title field)
        if title:
            title_lower = normalize_text(title.lower())
            title_found = False
            item_title = item.get("title", [])
            if isinstance(item_title, list):
                for t in item_title:
                    if isinstance(t, dict):
                        title_text = normalize_text(t.get("_value", "").lower())
                    else:
                        title_text = normalize_text(str(t).lower())
                    if title_lower in title_text:
                        title_found = True
                        break
            elif isinstance(item_title, str):
                if title_lower in normalize_text(item_title.lower()):
                    title_found = True
            if not title_found:
                continue

        # Check keyword filter (also searches in title and description)
        if keyword:
            keyword_lower = normalize_text(keyword.lower())
            found = False

            # Check keywords field
            item_keywords = item.get("keyword", [])
            if isinstance(item_keywords, str):
                item_keywords = [item_keywords]
            kw_texts = []
            for kw in item_keywords:
                if isinstance(kw, dict):
                    kw_texts.append(normalize_text(kw.get("_value", "").lower()))
                else:
                    kw_texts.append(normalize_text(str(kw).lower()))
            if keyword_lower in " ".join(kw_texts):
                found = True

            # Also check title
            if not found:
                item_title = item.get("title", [])
                if isinstance(item_title, list):
                    for t in item_title:
                        if isinstance(t, dict):
                            title_text = normalize_text(t.get("_value", "").lower())
                        else:
                            title_text = normalize_text(str(t).lower())
                        if keyword_lower in title_text:
                            found = True
                            break
                elif isinstance(item_title, str):
                    if keyword_lower in normalize_text(item_title.lower()):
                        found = True

            # Also check description
            if not found:
                item_desc = item.get("description", [])
                if isinstance(item_desc, list):
                    for d in item_desc:
                        if isinstance(d, dict):
                            desc_text = normalize_text(d.get("_value", "").lower())
                        else:
                            desc_text = normalize_text(str(d).lower())
                        if keyword_lower in desc_text:
                            found = True
                            break
                elif isinstance(item_desc, str):
                    if keyword_lower in normalize_text(item_desc.lower()):
                        found = True

            if not found:
                continue

        # Check license filter
        if license_filter:
            item_license = item.get("license", "")
            if isinstance(item_license, dict):
                item_license = item_license.get("_about", "") or item_license.get("_value", "")
            if license_filter.lower() not in str(item_license).lower():
                continue

        # Check frequency filter (accrualPeriodicity)
        if frequency_filter:
            item_freq = item.get("accrualPeriodicity", "")
            if isinstance(item_freq, dict):
                item_freq = item_freq.get("_about", "") or item_freq.get("_value", "")
            if frequency_filter.lower() not in str(item_freq).lower():
                continue

        filtered.append(item)

    return filtered


# Constants are now imported from core.config:
# DEFAULT_PAGE_SIZE, PREVIEW_MAX_BYTES, PREVIEW_TIMEOUT


async def _search_by_keywords_combined(
    keywords: list[str],
    pagination: PaginationParams,
) -> list[dict[str, Any]]:
    """Search datasets by multiple keywords using both keyword and title endpoints.

    Combines results from keyword search and title search to get comprehensive results.
    Deduplicates results by URI. Uses parallel requests for better performance.

    Args:
        keywords: List of keywords to search for.
        pagination: Pagination parameters.

    Returns:
        List of unique dataset items.
    """
    all_items: list[dict[str, Any]] = []
    existing_uris: set[str] = set()

    # Build list of coroutines for parallel execution
    async def search_keyword(kw: str) -> list[dict[str, Any]]:
        kw_normalized = normalize_text(kw)
        items = []
        try:
            kw_data = await client.get_datasets_by_keyword(kw_normalized, pagination)
            items.extend(kw_data.get("result", {}).get("items", []))
        except Exception as e:
            logger.warning("keyword_search_failed", keyword=kw_normalized, error=str(e))
        return items

    async def search_title(kw: str) -> list[dict[str, Any]]:
        kw_normalized = normalize_text(kw)
        items = []
        try:
            title_data = await client.search_datasets_by_title(kw_normalized, pagination)
            items.extend(title_data.get("result", {}).get("items", []))
        except Exception as e:
            logger.warning("title_search_failed", keyword=kw_normalized, error=str(e))
        return items

    # Execute all searches in parallel
    tasks = []
    for kw in keywords:
        tasks.append(search_keyword(kw))
        tasks.append(search_title(kw))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Deduplicate results
    for result in results:
        if isinstance(result, list):
            for item in result:
                uri = item.get("_about")
                if uri and uri not in existing_uris:
                    all_items.append(item)
                    existing_uris.add(uri)

    return all_items


async def _search_query_candidates(
    keywords: list[str],
    max_pages_per_source: int = 2,
    sort: str | None = None,
) -> list[dict[str, Any]]:
    """Busca candidatos usando keyword + title search en paralelo.

    Para cada keyword, busca en:
    - endpoint keyword (tags del dataset)
    - endpoint title (títulos del dataset)

    Ejecuta todas las búsquedas en paralelo con asyncio.gather y deduplica.

    Args:
        keywords: Lista de keywords extraídos de la query.
        max_pages_per_source: Máximo de páginas por endpoint (default 2).
        sort: Campo de ordenación.

    Returns:
        Lista de datasets únicos (deduplicados por URI).
    """
    seen_uris: set[str] = set()
    all_items: list[dict[str, Any]] = []

    async def fetch_pages(search_fn: Any, term: str) -> list[dict[str, Any]]:
        """Fetch múltiples páginas de un endpoint."""
        items: list[dict[str, Any]] = []
        for page in range(max_pages_per_source):
            pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE, sort=sort)
            try:
                data = await search_fn(term, pagination)
                page_items = data.get("result", {}).get("items", [])
                if not page_items:
                    break
                items.extend(page_items)
            except Exception as e:
                logger.warning("search_page_failed", term=term, page=page, error=str(e))
                break
        return items

    # Crear tareas para keyword y title search por cada keyword
    tasks = []
    for kw in keywords:
        kw_normalized = normalize_text(kw)
        tasks.append(fetch_pages(client.get_datasets_by_keyword, kw_normalized))
        tasks.append(fetch_pages(client.search_datasets_by_title, kw_normalized))

    # Ejecutar todas las búsquedas en paralelo
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Deduplicar resultados por URI
    for result in results:
        if isinstance(result, list):
            for item in result:
                uri = item.get("_about")
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    all_items.append(item)

    return all_items


def _normalize_preview_rows(preview_rows: int) -> int:
    """Normalize preview_rows to be within valid bounds (1-50)."""
    return min(max(1, preview_rows), 50)


def _build_filters_dict(**kwargs: Any) -> dict[str, Any]:
    """Build a dictionary of non-None filter values."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _validate_date(date_str: str | None, param_name: str) -> str | None:
    """Validate and normalize date string.

    Accepts formats:
    - ISO format: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SSZ
    - Compact format: YYYYMMDD

    Args:
        date_str: Date string to validate.
        param_name: Parameter name for error messages.

    Returns:
        Normalized date string (ISO format) or None if input is None.

    Raises:
        ValueError: If date format is invalid.
    """
    if not date_str:
        return None

    # Try ISO format first
    try:
        # Handle various ISO formats
        normalized = date_str.replace("Z", "+00:00")
        if "T" in normalized:
            from datetime import datetime

            datetime.fromisoformat(normalized)
        else:
            from datetime import datetime

            # Try YYYY-MM-DD
            datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        pass

    # Try compact format YYYYMMDD
    if len(date_str) == 8 and date_str.isdigit():
        try:
            from datetime import datetime

            dt = datetime.strptime(date_str, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    raise ValueError(
        f"Invalid {param_name} format: '{date_str}'. "
        f"Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ) or compact format (YYYYMMDD)."
    )


def _normalize_format(format_str: str | None, media_type: str | None) -> str | None:
    """Normalize format string to a standard format identifier."""
    if format_str:
        fmt_lower = format_str.lower()
        if "csv" in fmt_lower or fmt_lower == "text/csv" or "comma-separated" in fmt_lower:
            return "csv"
        if "json" in fmt_lower or fmt_lower == "application/json":
            return "json"
        if "tsv" in fmt_lower or "tab-separated" in fmt_lower:
            return "tsv"
    if media_type:
        mt_lower = media_type.lower()
        if "csv" in mt_lower:
            return "csv"
        if "json" in mt_lower:
            return "json"
        if "tab-separated" in mt_lower:
            return "tsv"
    return None


def _detect_csv_delimiter(content: str) -> str:
    """Detect the delimiter used in a CSV file."""
    # Check first few lines for common delimiters
    first_lines = content.split("\n")[:5]
    sample = "\n".join(first_lines)

    # Count occurrences of common delimiters
    delimiters = [",", ";", "\t", "|"]
    counts = {d: sample.count(d) for d in delimiters}

    # Return the most common delimiter (with at least 1 occurrence)
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _infer_value_type(value: Any) -> str:
    """Infer the data type of a single value."""
    if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
        return "null"

    if isinstance(value, bool):
        return "boolean"

    if isinstance(value, int):
        return "integer"

    if isinstance(value, float):
        return "float"

    if isinstance(value, str):
        val = value.strip()

        # Check for boolean strings
        if val.lower() in ("true", "false", "si", "no", "yes", "sí"):
            return "boolean"

        # Check for integer
        try:
            int(val.replace(",", "").replace(".", ""))
            if "." not in val and "," not in val:
                return "integer"
        except ValueError:
            pass

        # Check for float
        try:
            # Handle European format (1.234,56) and US format (1,234.56)
            normalized = val.replace(" ", "")
            if "," in normalized and "." in normalized:
                # Determine which is decimal separator
                if normalized.rfind(",") > normalized.rfind("."):
                    normalized = normalized.replace(".", "").replace(",", ".")
                else:
                    normalized = normalized.replace(",", "")
            elif "," in normalized:
                normalized = normalized.replace(",", ".")

            float(normalized)
            return "float"
        except ValueError:
            pass

        # Check for date patterns
        date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",  # 2024-01-15
            r"^\d{2}/\d{2}/\d{4}$",  # 15/01/2024
            r"^\d{2}-\d{2}-\d{4}$",  # 15-01-2024
        ]
        import re

        for pattern in date_patterns:
            if re.match(pattern, val):
                return "date"

        # Check for datetime patterns
        datetime_patterns = [
            r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",  # 2024-01-15T10:30 or 2024-01-15 10:30
        ]
        for pattern in datetime_patterns:
            if re.match(pattern, val):
                return "datetime"

        return "string"

    return "string"


def _calculate_column_stats(
    columns: list[str], rows: list[list[Any]], max_sample: int = 5
) -> list[ColumnStats]:
    """Calculate statistics for each column."""
    if not columns or not rows:
        return []

    stats: list[ColumnStats] = []

    for col_idx, col_name in enumerate(columns):
        # Extract column values
        values = []
        for row in rows:
            if col_idx < len(row):
                values.append(row[col_idx])
            else:
                values.append(None)

        # Infer types for each value
        types: dict[str, int] = {}
        null_count = 0
        non_null_values: list[Any] = []
        numeric_values: list[float] = []

        for val in values:
            val_type = _infer_value_type(val)
            types[val_type] = types.get(val_type, 0) + 1

            if val_type == "null":
                null_count += 1
            else:
                non_null_values.append(val)
                # Try to collect numeric values for min/max
                if val_type in ("integer", "float"):
                    try:
                        if isinstance(val, str):
                            normalized = val.replace(" ", "").replace(",", ".")
                            numeric_values.append(float(normalized))
                        else:
                            numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        pass

        # Determine overall type
        non_null_types = {t: c for t, c in types.items() if t != "null"}
        if not non_null_types:
            inferred_type = "null"
        elif len(non_null_types) == 1:
            inferred_type = list(non_null_types.keys())[0]
        else:
            # Mixed types - pick the most common non-null type
            most_common = max(non_null_types, key=non_null_types.get)
            if non_null_types[most_common] >= len(non_null_values) * 0.8:
                inferred_type = most_common
            else:
                inferred_type = "mixed"

        # Calculate unique count (only if reasonable number of values)
        unique_count = None
        if len(non_null_values) <= MAX_STATS_ROWS:
            try:
                unique_count = len(set(str(v) for v in non_null_values))
            except (TypeError, ValueError):
                pass

        # Get sample values
        sample_values = None
        if non_null_values:
            seen = set()
            samples = []
            for v in non_null_values:
                str_v = str(v)
                if str_v not in seen and len(samples) < max_sample:
                    samples.append(v)
                    seen.add(str_v)
            sample_values = samples if samples else None

        # Calculate min/max for numeric columns
        min_value = None
        max_value = None
        if numeric_values:
            min_value = min(numeric_values)
            max_value = max(numeric_values)
            # Round for display
            if min_value == int(min_value):
                min_value = int(min_value)
            if max_value == int(max_value):
                max_value = int(max_value)

        stats.append(
            ColumnStats(
                name=col_name,
                inferred_type=inferred_type,
                null_count=null_count,
                unique_count=unique_count,
                sample_values=sample_values,
                min_value=min_value,
                max_value=max_value,
            )
        )

    return stats


def _parse_csv_preview(content: str, max_rows: int) -> DataPreview:
    """Parse CSV content and return a preview with statistics."""
    try:
        delimiter = _detect_csv_delimiter(content)
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows_list = list(reader)

        if not rows_list:
            return DataPreview(
                columns=[],
                rows=[],
                total_rows=0,
                total_columns=0,
                format="csv",
                truncated=False,
            )

        columns = rows_list[0] if rows_list else []
        data_rows = rows_list[1 : max_rows + 1]
        total_rows = len(rows_list) - 1  # Exclude header

        # Calculate column statistics
        column_stats = _calculate_column_stats(columns, data_rows)

        return DataPreview(
            columns=columns,
            rows=data_rows,
            total_rows=total_rows,
            total_columns=len(columns),
            format="csv",
            truncated=len(rows_list) - 1 > max_rows,
            column_stats=column_stats if column_stats else None,
        )
    except csv.Error as e:
        return DataPreview(
            columns=[],
            rows=[],
            format="csv",
            error=f"CSV parsing error: {e}",
        )


def _parse_json_preview(content: str, max_rows: int) -> DataPreview:
    """Parse JSON content and return a preview with statistics."""
    try:
        data = json.loads(content)

        # Handle different JSON structures
        items: list[dict[str, Any]] = []

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try common patterns: data, items, results, records
            for key in ["data", "items", "results", "records", "rows"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            # If no list found, treat the dict itself as a single item
            if not items and data:
                items = [data]

        if not items:
            return DataPreview(
                columns=[],
                rows=[],
                total_rows=0,
                total_columns=0,
                format="json",
                truncated=False,
            )

        # Extract columns from first item
        first_item = items[0] if items else {}
        columns = list(first_item.keys()) if isinstance(first_item, dict) else []

        # Extract rows
        data_rows: list[list[Any]] = []
        for item in items[:max_rows]:
            if isinstance(item, dict):
                row = [item.get(col) for col in columns]
            else:
                row = [item]
            data_rows.append(row)

        # Calculate column statistics
        column_stats = _calculate_column_stats(columns, data_rows)

        return DataPreview(
            columns=columns,
            rows=data_rows,
            total_rows=len(items),
            total_columns=len(columns),
            format="json",
            truncated=len(items) > max_rows,
            column_stats=column_stats if column_stats else None,
        )
    except json.JSONDecodeError as e:
        return DataPreview(
            columns=[],
            rows=[],
            format="json",
            error=f"JSON parsing error: {e}",
        )


async def _fetch_data_preview(
    access_url: str,
    format_str: str | None,
    media_type: str | None,
    max_rows: int = 10,
) -> DataPreview | None:
    """Fetch and parse data preview from a distribution URL."""
    normalized_format = _normalize_format(format_str, media_type)

    if normalized_format not in ("csv", "json", "tsv"):
        return None  # Unsupported format

    file_size_bytes: int | None = None

    try:
        # Use follow_redirects to handle 301/302 redirects
        async with httpx.AsyncClient(timeout=PREVIEW_TIMEOUT, follow_redirects=True) as http_client:
            async with http_client.stream("GET", access_url) as response:
                response.raise_for_status()

                # Extract file size from Content-Length header
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        file_size_bytes = int(content_length)
                    except ValueError:
                        pass

                # Read up to PREVIEW_MAX_BYTES
                chunks = []
                bytes_read = 0
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    bytes_read += len(chunk)
                    if bytes_read >= PREVIEW_MAX_BYTES:
                        break

                content_bytes = b"".join(chunks)

                # Try to decode as UTF-8, fallback to latin-1
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = content_bytes.decode("latin-1", errors="replace")

        if normalized_format in ("csv", "tsv"):
            preview = _parse_csv_preview(content, max_rows)
            if normalized_format == "tsv":
                preview.format = "tsv"
            preview.file_size_bytes = file_size_bytes
            return preview
        elif normalized_format == "json":
            preview = _parse_json_preview(content, max_rows)
            preview.file_size_bytes = file_size_bytes
            return preview

    except httpx.TimeoutException:
        return DataPreview(
            columns=[],
            rows=[],
            format=normalized_format or "unknown",
            error="Timeout fetching data preview",
        )
    except httpx.HTTPStatusError as e:
        return DataPreview(
            columns=[],
            rows=[],
            format=normalized_format or "unknown",
            error=f"HTTP error {e.response.status_code}",
        )
    except Exception as e:
        return DataPreview(
            columns=[],
            rows=[],
            format=normalized_format or "unknown",
            error=f"Error fetching preview: {str(e)}",
        )

    return None


def _parse_csv_full(content: str) -> dict[str, Any]:
    """Parse full CSV content and return all rows with statistics."""
    try:
        delimiter = _detect_csv_delimiter(content)
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows_list = list(reader)

        if not rows_list:
            return {"columns": [], "rows": [], "total_rows": 0, "format": "csv"}

        columns = rows_list[0] if rows_list else []
        data_rows = rows_list[1:]

        # Calculate column statistics on sample (first MAX_STATS_ROWS rows)
        sample_rows = data_rows[:MAX_STATS_ROWS]
        column_stats = _calculate_column_stats(columns, sample_rows)

        return {
            "columns": columns,
            "rows": data_rows,
            "total_rows": len(data_rows),
            "total_columns": len(columns),
            "format": "csv",
            "column_stats": [s.model_dump() for s in column_stats] if column_stats else None,
        }
    except csv.Error as e:
        return {"error": f"CSV parsing error: {e}", "format": "csv"}


def _parse_json_full(content: str) -> dict[str, Any]:
    """Parse full JSON content and return all rows."""
    try:
        data = json.loads(content)

        # Handle different JSON structures
        items: list[dict[str, Any]] = []

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try common patterns
            for key in ["data", "items", "results", "records", "rows"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            if not items and data:
                items = [data]

        if not items:
            return {"columns": [], "rows": [], "total_rows": 0, "format": "json"}

        # Extract columns from first item
        first_item = items[0] if items else {}
        columns = list(first_item.keys()) if isinstance(first_item, dict) else []

        # Extract all rows
        data_rows: list[list[Any]] = []
        for item in items:
            if isinstance(item, dict):
                row = [item.get(col) for col in columns]
            else:
                row = [item]
            data_rows.append(row)

        # Calculate column statistics on sample
        sample_rows = data_rows[:MAX_STATS_ROWS]
        column_stats = _calculate_column_stats(columns, sample_rows)

        return {
            "columns": columns,
            "rows": data_rows,
            "total_rows": len(data_rows),
            "total_columns": len(columns),
            "format": "json",
            "column_stats": [s.model_dump() for s in column_stats] if column_stats else None,
        }
    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing error: {e}", "format": "json"}


async def _download_full_data(
    access_url: str,
    format_str: str | None,
    media_type: str | None,
    max_bytes: int = DOWNLOAD_MAX_BYTES,
    timeout: float = DOWNLOAD_TIMEOUT,
) -> dict[str, Any]:
    """Download and parse full data from a distribution URL.

    Args:
        access_url: URL to download from.
        format_str: Format string from distribution.
        media_type: Media type from distribution.
        max_bytes: Maximum bytes to download (default 50MB).
        timeout: Request timeout in seconds.

    Returns:
        Dict with columns, rows, total_rows, format, and optional error.
    """
    normalized_format = _normalize_format(format_str, media_type)

    if normalized_format not in ("csv", "json", "tsv"):
        return {"error": f"Unsupported format: {format_str}. Supported: csv, json, tsv"}

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http_client:
            async with http_client.stream("GET", access_url) as response:
                response.raise_for_status()

                # Get file size from header
                content_length = response.headers.get("content-length")
                file_size = int(content_length) if content_length else None

                chunks = []
                bytes_read = 0
                truncated = False

                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    bytes_read += len(chunk)
                    if bytes_read >= max_bytes:
                        truncated = True
                        break

                content_bytes = b"".join(chunks)

                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = content_bytes.decode("latin-1", errors="replace")

        # Parse based on format
        if normalized_format in ("csv", "tsv"):
            result = _parse_csv_full(content)
        else:
            result = _parse_json_full(content)

        result["file_size_bytes"] = file_size
        result["bytes_downloaded"] = bytes_read
        if truncated:
            result["truncated"] = True
            result["truncated_at_bytes"] = max_bytes

        return result

    except httpx.TimeoutException:
        return {"error": "Download timed out", "format": normalized_format or "unknown"}
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP error {e.response.status_code}",
            "format": normalized_format or "unknown",
        }
    except Exception as e:
        return {"error": str(e), "format": normalized_format or "unknown"}


async def _format_response_with_preview(data: dict[str, Any], preview_rows: int) -> str:
    """Format API response with data previews for distributions."""
    result = data.get("result", {})
    items = result.get("items", [])

    output: dict[str, Any] = {
        "total_in_page": len(items),
        "page": result.get("page", 0),
        "items_per_page": result.get("itemsPerPage", 10),
    }

    distributions = []
    for item in items:
        dist = DistributionSummary.from_api_item(item)

        # Try to fetch preview for supported formats
        if dist.access_url:
            preview = await _fetch_data_preview(
                dist.access_url,
                dist.format,
                dist.media_type,
                preview_rows,
            )
            if preview:
                dist.preview = preview

        distributions.append(dist.model_dump(exclude_none=True))

    output["distributions"] = distributions
    return json.dumps(output, ensure_ascii=False, indent=2)


async def _add_preview_to_dataset(
    dataset: DatasetSummary,
    item: dict[str, Any],
    max_rows: int = 10,
) -> DatasetSummary:
    """Add data preview to a dataset by fetching its first downloadable distribution.

    Prioritizes CSV and JSON formats for preview.
    """
    distributions = item.get("distribution", [])
    if isinstance(distributions, dict):
        distributions = [distributions]

    if not distributions:
        return dataset

    # Sort distributions to prioritize CSV, then JSON
    def format_priority(dist: dict) -> int:
        fmt = dist.get("format", "")
        if isinstance(fmt, dict):
            fmt = fmt.get("value", "") or fmt.get("_value", "")
        elif isinstance(fmt, list) and fmt:
            f = fmt[0]
            fmt = f.get("value", "") if isinstance(f, dict) else str(f)
        fmt_lower = str(fmt).lower()
        if "csv" in fmt_lower:
            return 0
        if "json" in fmt_lower:
            return 1
        return 99

    sorted_dists = sorted(distributions, key=format_priority)

    # Try to get preview from first suitable distribution
    for dist in sorted_dists:
        access_url = dist.get("accessURL")
        if not access_url:
            continue

        fmt = dist.get("format", "")
        if isinstance(fmt, dict):
            fmt = fmt.get("value", "") or fmt.get("_value", "")
        elif isinstance(fmt, list) and fmt:
            f = fmt[0]
            fmt = f.get("value", "") if isinstance(f, dict) else str(f)

        media_type = dist.get("mediaType", "")
        if isinstance(media_type, dict):
            media_type = media_type.get("value", "") or media_type.get("_value", "")

        # Check if format is supported for preview
        normalized = _normalize_format(fmt, media_type)
        if normalized in ("csv", "json", "tsv"):
            preview = await _fetch_data_preview(access_url, fmt, media_type, max_rows)
            if preview and not preview.error:
                dataset.preview = preview
                break

    return dataset


async def _add_previews_to_dataset_list(
    datasets: list[dict[str, Any]],
    preview_rows: int,
    fetch_distributions: bool = False,
) -> None:
    """Add data previews to a list of dataset dicts in-place.

    Args:
        datasets: List of dataset dictionaries to add previews to.
        preview_rows: Number of rows to include in preview.
        fetch_distributions: If True, fetch full dataset to get distributions
                            (needed for semantic search results).
    """
    preview_rows_limit = _normalize_preview_rows(preview_rows)

    for ds in datasets:
        if fetch_distributions:
            # For semantic search results, need to fetch full dataset
            uri = ds.get("uri", "")
            if not uri:
                continue
            try:
                dataset_id = uri.split("/")[-1]
                dataset_data = await client.get_dataset(dataset_id)
                dist_list = (
                    dataset_data.get("result", {}).get("primaryTopic", {}).get("distribution", [])
                )
                if isinstance(dist_list, dict):
                    dist_list = [dist_list]

                for dist_item in dist_list[:1]:  # Only first distribution
                    access_url = dist_item.get("accessURL")
                    if access_url:
                        fmt = dist_item.get("format", {})
                        if isinstance(fmt, dict):
                            fmt = fmt.get("_value", "")
                        preview = await _fetch_data_preview(
                            access_url,
                            fmt,
                            dist_item.get("mediaType"),
                            preview_rows_limit,
                        )
                        if preview and not preview.error:
                            ds["preview"] = preview.model_dump(exclude_none=True)
                            break
            except Exception as e:
                logger.warning("preview_fetch_failed", dataset_uri=ds.get("uri"), error=str(e))
        else:
            # For filter/hybrid results, distributions are already in the dataset
            distributions = ds.get("distributions", [])
            if not distributions:
                continue

            for dist in distributions:
                access_url = dist.get("access_url")
                if access_url:
                    preview = await _fetch_data_preview(
                        access_url,
                        dist.get("format"),
                        dist.get("media_type"),
                        preview_rows_limit,
                    )
                    if preview and not preview.error:
                        dist["preview"] = preview.model_dump(exclude_none=True)
                        break


async def _format_response_with_dataset_preview(
    data: dict[str, Any],
    lang: str | None = "es",
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
) -> str:
    """Format API response with data previews for datasets."""
    result = data.get("result", {})
    items = result.get("items", [])

    output: dict[str, Any] = {
        "total_in_page": len(items),
        "page": result.get("page", 0),
        "items_per_page": result.get("itemsPerPage", 10),
    }

    datasets = []
    for item in items:
        dataset = DatasetSummary.from_api_item(item, lang)
        dataset = await _add_preview_to_dataset(dataset, item, preview_rows)
        datasets.append(dataset.model_dump(exclude_none=True))

    output["datasets"] = datasets
    return json.dumps(output, ensure_ascii=False, indent=2)


# =============================================================================
# DATASET TOOLS
# =============================================================================


@mcp.tool()
async def get(
    dataset_id: str,
    include_data: bool = False,
    format: str | None = None,
    max_rows: int | None = None,
    max_mb: int = MAX_DOWNLOAD_MB,
    lang: str | None = "es",
) -> str:
    """Get dataset metadata and optionally download its data.

    Retrieve complete metadata for a dataset including all its distributions.
    Optionally download and parse the actual data.

    Args:
        dataset_id: The dataset identifier (slug from the URL or URI).
        include_data: If True, downloads and includes actual data (default: False).
        format: Preferred format when include_data=True ('csv', 'json'). Default: tries CSV first.
        max_rows: Maximum rows to return when include_data=True. None for all rows.
        max_mb: Maximum download size in MB (default 10, max 50).
        lang: Preferred language ('es', 'en', 'ca', 'eu', 'gl'). Default 'es'.

    Returns:
        JSON with dataset metadata. If include_data=True, also includes columns and rows.
    """
    usage_metrics.record_tool_call("get")
    usage_metrics.record_dataset_access(dataset_id)

    try:
        data = await client.get_dataset(dataset_id)

        if not include_data:
            return _format_response(data, "dataset", lang)

        # Download data logic (from download_data)
        max_mb = min(max_mb, MAX_DOWNLOAD_MB)  # Safety limit
        distributions = data.get("result", {}).get("primaryTopic", {}).get("distribution", [])

        if isinstance(distributions, dict):
            distributions = [distributions]

        if not distributions:
            return json.dumps(
                {"error": "No distributions found for this dataset"}, ensure_ascii=False
            )

        # Find best distribution matching preferred format
        best_dist = None
        for dist in distributions:
            dist_format = dist.get("format", "")
            if isinstance(dist_format, dict):
                dist_format = dist_format.get("_value", "")

            normalized = _normalize_format(dist_format, dist.get("mediaType"))

            if normalized in ("csv", "json", "tsv"):
                if format:
                    if normalized == format.lower():
                        best_dist = dist
                        break
                elif not best_dist:
                    best_dist = dist
                    if normalized == "csv":
                        break

        if not best_dist:
            available_formats = [
                d.get("format", {}).get("_value", d.get("format", "unknown")) for d in distributions
            ]
            return json.dumps(
                {
                    "error": f"No compatible distribution found. Available formats: {available_formats}",
                    "supported_formats": ["csv", "json", "tsv"],
                },
                ensure_ascii=False,
            )

        access_url = best_dist.get("accessURL")
        if not access_url:
            return json.dumps({"error": "Distribution has no access URL"}, ensure_ascii=False)

        # Download and parse
        result = await _download_full_data(
            access_url,
            best_dist.get("format"),
            best_dist.get("mediaType"),
            max_bytes=max_mb * 1024 * 1024,
        )

        # Truncate rows if max_rows specified
        if max_rows and "rows" in result:
            original_rows = len(result["rows"])
            result["rows"] = result["rows"][:max_rows]
            if original_rows > max_rows:
                result["rows_truncated_to"] = max_rows
                result["total_rows_available"] = original_rows

        result["dataset_id"] = dataset_id
        result["source_url"] = access_url

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return _handle_error(e)


async def _search_datasets_impl(
    query: str | None = None,
    title: str | None = None,
    publisher: str | None = None,
    theme: str | None = None,
    themes: list[str] | None = None,
    format: str | None = None,
    keyword: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    exact_match: bool = False,
    page: int = 0,
    sort: str | None = None,
    lang: str | None = "es",
    fetch_all: bool = False,
    max_results: int = MAX_SEARCH_RESULTS,
    include_preview: bool = False,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    semantic_min_score: float = SEMANTIC_MIN_SCORE,
    license: str | None = None,
    frequency: str | None = None,
) -> str:
    """Internal implementation of search_datasets.

    Supports two search modes:
    1. Query-based: Natural language query → extract keywords → search API → semantic re-ranking.
    2. Filter-based: Use title, keyword, theme, publisher, etc. for direct API filtering.

    Response includes enriched metadata: id, publisher_name, frequency, language,
    spatial, license, formats, and access_url.

    Args:
        query: Natural language query (e.g., "accidentes de tráfico en Madrid"). Extracts keywords, searches API, then re-ranks by semantic similarity.
        title: Search text in dataset titles (direct API search, no semantic re-ranking).
        publisher: Publisher ID (e.g., 'EA0010587' for INE). Use list_metadata('publishers') to find IDs.
        theme: Theme ID (e.g., 'economia', 'salud', 'educacion'). Use list_metadata('themes') to find IDs.
        themes: List of theme IDs for multi-theme search (OR logic). Example: ['economia', 'hacienda'].
        format: Format ID (e.g., 'csv', 'json', 'xml').
        keyword: Keyword/tag to filter by (e.g., 'presupuesto', 'poblacion').
        date_start: Start date 'YYYY-MM-DDTHH:mmZ' (e.g., '2024-01-01T00:00Z').
        date_end: End date 'YYYY-MM-DDTHH:mmZ' (e.g., '2024-12-31T23:59Z').
        exact_match: If True with title, match whole words only.
        page: Page number (starting from 0). Ignored if fetch_all=True.
        sort: Sort field. Use '-' prefix for descending. Examples: '-modified', 'title'.
        lang: Preferred language ('es', 'en', 'ca', 'eu', 'gl'). Default 'es'.
        fetch_all: If True, fetches all pages automatically up to max_results.
        max_results: Maximum results (default 100, max 100).
        include_preview: Include data preview for CSV/JSON/TSV datasets. Default False.
        preview_rows: Number of preview rows (default 10, max 50).
        semantic_min_score: Min similarity score 0-1 for query results (default 0.5).
        license: Filter by license type (e.g., 'CC_BY', 'CC0').
        frequency: Filter by update frequency ('P1D', 'P1W', 'P1M', 'P1Y').

    Returns:
        JSON with matching datasets including metadata.
    """
    # Record usage metrics
    usage_metrics.record_tool_call("search_datasets")
    usage_metrics.record_search(
        {
            "title": title,
            "publisher": publisher,
            "theme": theme,
            "themes": themes,
            "format": format,
            "keyword": keyword,
            "query": query,
        }
    )

    try:
        # Validate date formats first
        date_start = _validate_date(date_start, "date_start")
        date_end = _validate_date(date_end, "date_end")

        max_results = min(max_results, MAX_SEARCH_RESULTS)  # Safety limit
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE, sort=sort)
        data: dict[str, Any] | None = None
        local_filters: dict[str, Any] = {}

        # =================================================================
        # QUERY-BASED SEARCH (keywords + semantic re-ranking)
        # =================================================================
        if query:
            # Step 1: Extract keywords from natural language query
            keywords = _extract_keywords(query)

            if not keywords:
                return json.dumps(
                    {
                        "error": "No se pudieron extraer keywords de la query",
                        "query": query,
                        "suggestion": "Usa términos más específicos o usa el parámetro 'title' para búsqueda directa.",
                    },
                    ensure_ascii=False,
                )

            # =============================================================
            # STEP 1: KEYWORD + TITLE SEARCH - Búsqueda amplia en paralelo
            # =============================================================
            # Buscamos en endpoints keyword y title para cada keyword extraído
            # Ejecuta todas las búsquedas en paralelo y deduplica resultados
            keywords_to_search = keywords[:5]
            all_items = await _search_query_candidates(
                keywords=keywords_to_search,
                max_pages_per_source=2,
                sort=sort,
            )

            # Aplicar filtros tradicionales si se proporcionaron
            unique_items = all_items
            if any([publisher, theme, themes, format, date_start]):
                unique_items = _filter_datasets_locally(
                    unique_items,
                    publisher,
                    theme,
                    themes,
                    format,
                    keyword=None,
                    license_filter=license,
                    frequency_filter=frequency,
                )

            total_from_search = len(unique_items)

            if not unique_items:
                return json.dumps(
                    {
                        "search_mode": "query (keyword + title search)",
                        "query": query,
                        "keywords_extracted": keywords_to_search,
                        "step1_search": {
                            "sources": ["keyword", "title"],
                            "total_results": 0,
                        },
                        "total_results": 0,
                        "datasets": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # Construir summaries para el pool de candidatos (máx 500)
            max_candidates = MAX_SEMANTIC_CANDIDATES
            datasets = []
            for item in unique_items[:max_candidates]:
                summary = DatasetSummary.from_api_item(item, lang)
                datasets.append(summary.model_dump(exclude_none=True))

            # =============================================================
            # STEP 2: SEMANTIC FILTER - Filtrar por similaridad semántica
            # =============================================================
            if not _load_embeddings_dependencies():
                # Fallback: devolver resultados sin filtro semántico
                datasets = datasets[:max_results]
                return json.dumps(
                    {
                        "search_mode": "query (keyword + title search, no semantic filter)",
                        "query": query,
                        "keywords_extracted": keywords_to_search,
                        "step1_search": {
                            "sources": ["keyword", "title"],
                            "total_raw_results": len(all_items),
                            "total_after_filters": total_from_search,
                        },
                        "total_results": len(datasets),
                        "note": "Semantic filter not available (install sentence-transformers)",
                        "datasets": datasets,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # Aplicar filtro semántico: solo mantener resultados con score >= min_score
            ranked = semantic_ranker.rank_results(
                query, datasets, min_score=semantic_min_score, top_k=max_results
            )

            # Añadir previews si se solicitaron
            if include_preview and ranked:
                await _add_previews_to_dataset_list(ranked, preview_rows)

            return json.dumps(
                {
                    "search_mode": "query (keyword + title search + semantic filter)",
                    "query": query,
                    "keywords_extracted": keywords_to_search,
                    "step1_search": {
                        "sources": ["keyword", "title"],
                        "total_raw_results": len(all_items),
                        "total_after_filters": total_from_search,
                        "candidates_for_semantic": len(datasets),
                    },
                    "step2_semantic_filter": {
                        "min_score": semantic_min_score,
                        "results_passing_filter": len(ranked),
                    },
                    "total_results": len(ranked),
                    "datasets": ranked,
                },
                ensure_ascii=False,
                indent=2,
            )

        # =================================================================
        # FILTER-BASED SEARCH MODE (traditional, no semantic re-ranking)
        # =================================================================
        # Priority order for API query: title > date > publisher > theme > format > keyword
        if title:
            if exact_match:
                # Search multiple pages to find exact matches
                max_pages = 50 if fetch_all else 10
                all_filtered_items: list[dict[str, Any]] = []

                # Normalize title for API (remove accents)
                normalized_title = normalize_text(title)
                for current_page in range(max_pages):
                    page_pagination = PaginationParams(
                        page=current_page, page_size=DEFAULT_PAGE_SIZE, sort=sort
                    )
                    data = await client.search_datasets_by_title(normalized_title, page_pagination)

                    result = data.get("result", {})
                    items = result.get("items", [])

                    if not items:
                        break

                    filtered = _filter_items_by_exact_match(items, title)
                    # Apply local filters
                    filtered = _filter_datasets_locally(
                        filtered,
                        publisher,
                        theme,
                        themes,
                        format,
                        keyword,
                        license_filter=license,
                        frequency_filter=frequency,
                    )
                    all_filtered_items.extend(filtered)

                    if not fetch_all and len(all_filtered_items) >= DEFAULT_PAGE_SIZE:
                        break
                    if fetch_all and len(all_filtered_items) >= max_results:
                        break

                if fetch_all:
                    all_filtered_items = all_filtered_items[:max_results]
                    items_to_process = all_filtered_items
                    output = {
                        "total_results": len(all_filtered_items),
                        "fetch_all": True,
                        "exact_match": True,
                    }
                else:
                    start_idx = page * DEFAULT_PAGE_SIZE
                    end_idx = start_idx + DEFAULT_PAGE_SIZE
                    items_to_process = all_filtered_items[start_idx:end_idx]
                    output = {
                        "total_in_page": len(items_to_process),
                        "total_exact_matches": len(all_filtered_items),
                        "page": page,
                        "items_per_page": DEFAULT_PAGE_SIZE,
                        "exact_match": True,
                    }

                # Build datasets with optional preview
                datasets = []
                preview_rows_limit = min(max(1, preview_rows), 50) if include_preview else 0
                for item in items_to_process:
                    dataset = DatasetSummary.from_api_item(item, lang)
                    if include_preview:
                        dataset = await _add_preview_to_dataset(dataset, item, preview_rows_limit)
                    datasets.append(dataset.model_dump(exclude_none=True))
                output["datasets"] = datasets

                return json.dumps(output, ensure_ascii=False, indent=2)
            else:
                # Normalize title for API (remove accents)
                normalized_title = normalize_text(title)
                data = await client.search_datasets_by_title(normalized_title, pagination)
                # keyword filtering is handled locally by _filter_datasets_locally
                local_filters = {
                    "publisher": publisher,
                    "theme": theme,
                    "themes": themes,
                    "format": format,
                    "keyword": keyword,
                    "title": title,
                }

        elif date_start and date_end:
            data = await client.get_datasets_by_date_range(date_start, date_end, pagination)
            local_filters = {
                "publisher": publisher,
                "theme": theme,
                "themes": themes,
                "format": format,
                "keyword": keyword,
            }

        elif publisher:
            data = await client.get_datasets_by_publisher(publisher, pagination)
            local_filters = {"theme": theme, "themes": themes, "format": format, "keyword": keyword}

        elif theme or themes:
            # If single theme provided, use API; if multiple, fetch all and filter
            if theme and not themes:
                data = await client.get_datasets_by_theme(theme, pagination)
                local_filters = {"publisher": publisher, "format": format, "keyword": keyword}
            else:
                # Multiple themes - need to fetch and filter locally
                data = await client.list_datasets(pagination)
                local_filters = {
                    "publisher": publisher,
                    "theme": theme,
                    "themes": themes,
                    "format": format,
                    "keyword": keyword,
                }

        elif format:
            data = await client.get_datasets_by_format(format, pagination)
            local_filters = {
                "publisher": publisher,
                "theme": theme,
                "themes": themes,
                "keyword": keyword,
            }

        elif keyword:
            # Search by keyword AND by title to get comprehensive results
            if fetch_all:
                # Fetch multiple pages for keyword search
                all_items: list[dict[str, Any]] = []
                existing_uris: set[str] = set()
                current_page = 0
                max_pages = 20  # Limit to prevent excessive API calls

                while len(all_items) < max_results and current_page < max_pages:
                    page_pagination = PaginationParams(
                        page=current_page, page_size=DEFAULT_PAGE_SIZE, sort=sort
                    )
                    page_items = await _search_by_keywords_combined([keyword], page_pagination)

                    if not page_items:
                        break

                    # Deduplicate
                    new_items = 0
                    for item in page_items:
                        uri = item.get("_about")
                        if uri and uri not in existing_uris:
                            all_items.append(item)
                            existing_uris.add(uri)
                            new_items += 1

                    # Stop if no new items found
                    if new_items == 0:
                        break

                    current_page += 1

                all_items = all_items[:max_results]
            else:
                all_items = await _search_by_keywords_combined([keyword], pagination)

            data = {
                "result": {
                    "items": all_items,
                    "page": 0,
                    "itemsPerPage": len(all_items),
                    "fetch_all": fetch_all,
                }
            }
            local_filters = {
                "publisher": publisher,
                "theme": theme,
                "themes": themes,
                "format": format,
            }

        else:
            # No filters provided, list all datasets
            data = await client.list_datasets(pagination)

        # Apply local filtering if needed
        if data and any(local_filters.values()):
            result = data.get("result", {})
            items = result.get("items", [])
            filtered_items = _filter_datasets_locally(
                items,
                local_filters.get("publisher"),
                local_filters.get("theme"),
                local_filters.get("themes"),
                local_filters.get("format"),
                local_filters.get("keyword"),
                local_filters.get("title"),
                license_filter=license,
                frequency_filter=frequency,
            )
            data["result"]["items"] = filtered_items

        if not data:
            return json.dumps({"error": "No data"})

        # Return with preview if requested
        if include_preview:
            preview_rows = min(max(1, preview_rows), 50)  # Limit to 1-50 rows
            return await _format_response_with_dataset_preview(data, lang, preview_rows)

        return _format_response(data, "dataset", lang)

    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def search(
    query: str | None = None,
    title: str | None = None,
    publisher: str | None = None,
    theme: str | None = None,
    themes: list[str] | None = None,
    format: str | None = None,
    keyword: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    exact_match: bool = False,
    page: int = 0,
    sort: str | None = None,
    lang: str | None = "es",
    fetch_all: bool = False,
    max_results: int = MAX_SEARCH_RESULTS,
    include_preview: bool = False,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    semantic_min_score: float = SEMANTIC_MIN_SCORE,
    license: str | None = None,
    frequency: str | None = None,
) -> str:
    """Search datasets from the Spanish open data catalog (datos.gob.es).

    Two search modes:
    1. Query-based: Natural language query → extracts keywords → searches API → re-ranks by semantic similarity.
    2. Filter-based: Direct API filtering by title, publisher, theme, etc.

    Args:
        query: Natural language query (e.g., "accidentes de tráfico en Madrid").
               Extracts keywords, searches API, then re-ranks results by semantic similarity.
        title: Search text in dataset titles (direct API search, no semantic re-ranking).
        publisher: Publisher ID (e.g., 'EA0010587' for INE, 'L01280796' for Ayto. Madrid).
        theme: Theme ID (e.g., 'economia', 'salud', 'medio-ambiente', 'transporte').
        themes: List of theme IDs for multi-theme search (OR logic).
        format: Format ID (e.g., 'csv', 'json', 'xml').
        keyword: Keyword/tag to filter by (e.g., 'presupuesto').
        date_start: Start date 'YYYY-MM-DDTHH:mmZ'.
        date_end: End date 'YYYY-MM-DDTHH:mmZ'.
        exact_match: If True with title, match whole words only.
        page: Page number (starting from 0).
        sort: Sort field. Use '-' prefix for descending (e.g., '-modified').
        lang: Preferred language ('es', 'en', 'ca', 'eu', 'gl'). Default 'es'.
        fetch_all: If True, fetches all pages up to max_results.
        max_results: Maximum results (default 100, max 100).
        include_preview: Include data preview for CSV/JSON datasets. Default False.
        preview_rows: Number of preview rows (default 10, max 50).
        semantic_min_score: Min similarity score 0-1 for query results (default 0.5).
        license: Filter by license type (e.g., 'CC_BY', 'CC0').
        frequency: Filter by update frequency ('P1D', 'P1W', 'P1M', 'P1Y').

    Returns:
        JSON with matching datasets and metadata.
    """
    return await _search_datasets_impl(
        query=query,
        title=title,
        publisher=publisher,
        theme=theme,
        themes=themes,
        format=format,
        keyword=keyword,
        date_start=date_start,
        date_end=date_end,
        exact_match=exact_match,
        page=page,
        sort=sort,
        lang=lang,
        fetch_all=fetch_all,
        max_results=max_results,
        include_preview=include_preview,
        preview_rows=preview_rows,
        semantic_min_score=semantic_min_score,
        license=license,
        frequency=frequency,
    )


# =============================================================================
# RESOURCE TEMPLATES - Dynamic Resources
# =============================================================================


@mcp.resource("dataset://{dataset_id}")
async def resource_dataset(dataset_id: str) -> str:
    """
    Acceso directo a un dataset específico por su identificador.

    Este recurso permite obtener toda la información de un dataset concreto,
    incluyendo:
    - Metadatos completos (título, descripción, fecha de publicación)
    - Información del publicador
    - Temáticas y palabras clave asociadas
    - Lista de distribuciones (archivos descargables)
    - URLs de acceso a los datos
    - Frecuencia de actualización
    - Cobertura temporal y geográfica

    El dataset_id es el identificador único del dataset, que puede obtenerse
    de las búsquedas o del URI del dataset en datos.gob.es.

    Ejemplo: dataset://ea0010587-poblacion-por-sexo-municipios-y-edad
    """
    try:
        data = await client.get_dataset(dataset_id)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.resource("theme://{theme_id}")
async def resource_theme_datasets(theme_id: str) -> str:
    """
    Acceso directo a datasets de una temática específica.

    Este recurso permite obtener todos los datasets clasificados bajo una
    temática determinada del catálogo de datos.gob.es.

    Temáticas disponibles:
    - economia: Datos económicos, PIB, comercio exterior, empresas
    - hacienda: Presupuestos públicos, impuestos, deuda, gasto
    - educacion: Sistema educativo, universidades, becas, alumnado
    - salud: Sanidad, hospitales, epidemiología, farmacia
    - medio-ambiente: Calidad del aire, agua, residuos, biodiversidad
    - transporte: Movilidad, carreteras, ferrocarril, aviación
    - turismo: Visitantes, alojamientos, destinos turísticos
    - empleo: Mercado laboral, paro, contratación, salarios
    - sector-publico: Administración, funcionarios, organismos
    - ciencia-tecnologia: I+D, innovación, patentes, startups
    - cultura-ocio: Museos, bibliotecas, espectáculos
    - urbanismo-infraestructuras: Vivienda, construcción, catastro
    - energia: Electricidad, gas, renovables, consumo energético

    Ejemplo: theme://economia
    """
    try:
        pagination = PaginationParams(page=0, page_size=DEFAULT_PAGE_SIZE)
        data = await client.get_datasets_by_theme(theme_id, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.resource("publisher://{publisher_id}")
async def resource_publisher_datasets(publisher_id: str) -> str:
    """
    Acceso directo a datasets de un publicador específico.

    Este recurso permite obtener todos los datasets publicados por un
    organismo o institución determinada.

    Publicadores principales:
    - EA0010587: INE (Instituto Nacional de Estadística)
    - E05024401: Ministerio de Hacienda
    - E05024301: Ministerio de Economía
    - E00003901: Agencia Estatal de Meteorología (AEMET)
    - A08002970: Generalitat de Catalunya
    - A01002820: Gobierno Vasco
    - A13002908: Junta de Andalucía
    - L01280796: Ayuntamiento de Madrid
    - L01080193: Ajuntament de Barcelona

    Para obtener la lista completa de publicadores, consulta el recurso
    catalog://publishers.

    El publisher_id corresponde al código DIR3 del organismo o su
    identificador en el catálogo.

    Ejemplo: publisher://EA0010587
    """
    try:
        pagination = PaginationParams(page=0, page_size=DEFAULT_PAGE_SIZE)
        data = await client.get_datasets_by_publisher(publisher_id, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.resource("format://{format_id}")
async def resource_format_datasets(format_id: str) -> str:
    """
    Acceso directo a datasets disponibles en un formato específico.

    Este recurso permite obtener datasets que tienen distribuciones
    (archivos descargables) en el formato solicitado.

    Formatos disponibles:
    - csv: Comma-Separated Values (ideal para análisis de datos)
    - json: JavaScript Object Notation (ideal para APIs y aplicaciones)
    - xml: eXtensible Markup Language (interoperabilidad)
    - xlsx: Microsoft Excel (hojas de cálculo)
    - xls: Microsoft Excel antiguo
    - rdf: Resource Description Framework (datos enlazados/linked data)
    - pdf: Portable Document Format (documentos)
    - zip: Archivos comprimidos
    - shp: Shapefile (datos geográficos)
    - geojson: GeoJSON (datos geográficos)
    - kml: Keyhole Markup Language (Google Earth)
    - html: Páginas web
    - api: Acceso mediante API

    Ejemplo: format://csv
    """
    try:
        pagination = PaginationParams(page=0, page_size=DEFAULT_PAGE_SIZE)
        data = await client.get_datasets_by_format(format_id, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.resource("keyword://{keyword}")
async def resource_keyword_datasets(keyword: str) -> str:
    """
    Acceso directo a datasets etiquetados con una palabra clave específica.

    Este recurso permite obtener datasets que han sido etiquetados con
    un keyword o tag determinado. Los keywords son más específicos que
    las temáticas y permiten búsquedas más precisas.

    Keywords populares:
    - presupuesto, gastos, ingresos (finanzas públicas)
    - poblacion, censo, demografia (estadísticas poblacionales)
    - empleo, paro, contratacion (mercado laboral)
    - covid, pandemia, salud (sanidad)
    - clima, temperatura, precipitacion (meteorología)
    - transporte, trafico, movilidad (transporte)
    - educacion, universidad, alumnos (educación)
    - turismo, viajeros, hoteles (turismo)
    - energia, electricidad, consumo (energía)
    - medioambiente, contaminacion, residuos (medio ambiente)

    Los keywords pueden estar en español o inglés según el dataset.

    Ejemplo: keyword://presupuesto
    """
    try:
        pagination = PaginationParams(page=0, page_size=DEFAULT_PAGE_SIZE)
        # Normalize keyword for API (remove accents)
        normalized_keyword = normalize_text(keyword)
        data = await client.get_datasets_by_keyword(normalized_keyword, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


# =============================================================================
# PROMPTS - Guided Search Workflows (loaded from prompts/ directory)
# =============================================================================

# Import and register prompts from the prompts package
try:
    from prompts import register_prompts

    register_prompts(mcp)
except ImportError:
    # Prompts folder not available, skip registration
    pass


# =============================================================================
# INTEGRATIONS - External APIs (INE, AEMET, BOE)
# =============================================================================

# Import and register INE tools
try:
    from integrations.ine import register_ine_tools

    register_ine_tools(mcp)
except ImportError:
    # INE integration not available
    pass

# Import and register AEMET tools
try:
    from integrations.aemet import register_aemet_tools

    register_aemet_tools(mcp)
except ImportError:
    # AEMET integration not available
    pass

# Import and register BOE tools
try:
    from integrations.boe import register_boe_tools

    register_boe_tools(mcp)
except ImportError:
    # BOE integration not available
    pass


# Export mcp for FastMCP Cloud
# The 'mcp' object is the FastMCP server instance that FastMCP Cloud will use

if __name__ == "__main__":
    mcp.run()
