"""MCP server for datos.gob.es open data catalog API."""

import csv
import io
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

from urllib.parse import urljoin

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Core utilities for logging and rate limiting
from core import setup_logging, get_logger, HTTPClient

# Initialize structured logging
setup_logging()
logger = get_logger("datos_gob_es")

# Optional imports for semantic search (lazy loaded)
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    np = None
    SentenceTransformer = None


# =============================================================================
# MODELS
# =============================================================================


class PaginationParams(BaseModel):
    """Parameters for paginated API requests."""

    page: int = Field(default=0, ge=0, description="Page number (0-indexed)")
    page_size: int = Field(default=200, ge=1, le=200, description="Items per page (max 200)")
    sort: str | None = Field(
        default=None, description="Sort field(s), prefix with - for descending"
    )


class DatasetSummary(BaseModel):
    """Simplified dataset representation for responses."""

    uri: str
    title: str | list[str] | None = None
    description: str | list[str] | None = None
    publisher: str | dict[str, Any] | None = None
    theme: list[str] | str | None = None
    keywords: list[str] | None = None
    issued: str | None = None
    modified: str | None = None
    distributions_count: int = 0

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

        distributions = item.get("distribution", [])
        if isinstance(distributions, dict):
            distributions = [distributions]

        theme = item.get("theme")
        if isinstance(theme, str):
            theme = [theme]
        elif isinstance(theme, list):
            theme = [t if isinstance(t, str) else str(t) for t in theme]

        keywords = cls._extract_keywords(item.get("keyword", []), lang)

        return cls(
            uri=item.get("_about", ""),
            title=title,
            description=description,
            publisher=item.get("publisher"),
            theme=theme,
            keywords=keywords,
            issued=item.get("issued"),
            modified=item.get("modified"),
            distributions_count=len(distributions) if distributions else 0,
        )

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


class DataPreview(BaseModel):
    """Preview of actual data content from a distribution."""

    columns: list[str]
    rows: list[list[Any]]
    total_rows: int | None = None
    format: str
    truncated: bool = False
    error: str | None = None


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
# EMBEDDING INDEX FOR SEMANTIC SEARCH
# =============================================================================


class EmbeddingIndex:
    """Manages dataset embeddings for semantic search.

    Provides lazy-loaded semantic search capabilities using sentence-transformers.
    The embedding index is cached to disk for faster subsequent loads.
    """

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CACHE_DIR = Path.home() / ".cache" / "datos-gob-es"
    CACHE_FILE = CACHE_DIR / "embeddings.pkl"

    def __init__(self):
        self.model = None
        self.embeddings = None  # numpy array of shape (n_datasets, embedding_dim)
        self.dataset_ids: list[str] = []
        self.dataset_titles: list[str] = []
        self.dataset_descriptions: list[str] = []
        self._initialized = False

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Lazy load the embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            raise RuntimeError(
                "Semantic search requires sentence-transformers and numpy. "
                "Install with: pip install sentence-transformers numpy"
            )
        if self.model is None:
            self.model = SentenceTransformer(self.MODEL_NAME)

    def _load_cache(self) -> bool:
        """Load embeddings from cache file if it exists.

        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        if not self.CACHE_FILE.exists():
            return False

        try:
            with open(self.CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)

            self.embeddings = cache_data["embeddings"]
            self.dataset_ids = cache_data["dataset_ids"]
            self.dataset_titles = cache_data["dataset_titles"]
            self.dataset_descriptions = cache_data.get("dataset_descriptions", [])
            self._initialized = True
            return True
        except Exception:
            return False

    def _save_cache(self):
        """Save embeddings to cache file."""
        self._ensure_cache_dir()
        cache_data = {
            "embeddings": self.embeddings,
            "dataset_ids": self.dataset_ids,
            "dataset_titles": self.dataset_titles,
            "dataset_descriptions": self.dataset_descriptions,
        }
        with open(self.CACHE_FILE, "wb") as f:
            pickle.dump(cache_data, f)

    async def build_index(self, client: "DatosGobClient", max_datasets: int = 5000):
        """Build embedding index from the catalog.

        Args:
            client: The API client to fetch datasets.
            max_datasets: Maximum number of datasets to index.
        """
        self._load_model()

        # Fetch all datasets
        all_items: list[dict[str, Any]] = []
        page = 0

        while len(all_items) < max_datasets:
            pagination = PaginationParams(page=page, page_size=200)
            data = await client.list_datasets(pagination)

            result = data.get("result", {})
            items = result.get("items", [])

            if not items:
                break

            all_items.extend(items)

            if len(items) < 200:
                break

            page += 1

        all_items = all_items[:max_datasets]

        # Extract text for each dataset
        self.dataset_ids = []
        self.dataset_titles = []
        self.dataset_descriptions = []
        texts_to_encode: list[str] = []

        for item in all_items:
            dataset_id = item.get("_about", "")
            title = DatasetSummary._extract_text(item.get("title"), "es")
            description = DatasetSummary._extract_text(item.get("description"), "es")

            # Convert to string
            title_str = title if isinstance(title, str) else (title[0] if title else "")
            desc_str = description if isinstance(description, str) else (description[0] if description else "")

            self.dataset_ids.append(dataset_id)
            self.dataset_titles.append(title_str)
            self.dataset_descriptions.append(desc_str)

            # Combine title and description for embedding
            combined_text = f"{title_str}. {desc_str}" if desc_str else title_str
            texts_to_encode.append(combined_text)

        # Generate embeddings
        self.embeddings = self.model.encode(texts_to_encode, show_progress_bar=False)
        self._initialized = True

        # Save to cache
        self._save_cache()

    def search(self, query: str, top_k: int = 20, min_score: float = 0.3) -> list[dict[str, Any]]:
        """Search for similar datasets using cosine similarity.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity score (0-1).

        Returns:
            List of dicts with dataset_id, title, description, and score.
        """
        if not self._initialized:
            raise RuntimeError("Index not initialized. Call build_index() first.")

        self._load_model()

        # Encode query
        query_embedding = self.model.encode(query)

        # Calculate cosine similarity
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                break
            results.append({
                "dataset_id": self.dataset_ids[idx],
                "title": self.dataset_titles[idx],
                "description": self.dataset_descriptions[idx][:200] + "..." if len(self.dataset_descriptions[idx]) > 200 else self.dataset_descriptions[idx],
                "score": round(score, 4),
            })

        return results

    def clear_cache(self):
        """Delete the cache file."""
        if self.CACHE_FILE.exists():
            self.CACHE_FILE.unlink()
        self._initialized = False
        self.embeddings = None
        self.dataset_ids = []
        self.dataset_titles = []
        self.dataset_descriptions = []


# Global embedding index instance
embedding_index = EmbeddingIndex()


# =============================================================================
# HTTP CLIENT
# =============================================================================


class DatosGobClientError(Exception):
    """Exception raised for API client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DatosGobClient:
    """Async HTTP client for the datos.gob.es API.

    Uses HTTPClient for automatic logging and rate limiting.
    """

    BASE_URL = "https://datos.gob.es/apidata/"
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.http = HTTPClient("datos.gob.es", self.BASE_URL, timeout)

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
            if hasattr(e, 'status_code'):
                raise DatosGobClientError(str(e), status_code=e.status_code) from e
            raise DatosGobClientError(str(e)) from e

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

    async def list_distributions(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
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

    async def list_spatial_coverage(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("catalog/spatial", params)

    async def list_themes(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("catalog/theme", params)

    async def list_public_sectors(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("nti/public-sector", params)

    async def get_public_sector(self, sector_id: str) -> dict[str, Any]:
        return await self._request(f"nti/public-sector/{sector_id}")

    async def list_provinces(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("nti/territory/Province", params)

    async def get_province(self, province_id: str) -> dict[str, Any]:
        return await self._request(f"nti/territory/Province/{province_id}")

    async def list_autonomous_regions(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        params = self._build_params(pagination)
        return await self._request("nti/territory/Autonomous-region", params)

    async def get_autonomous_region(self, region_id: str) -> dict[str, Any]:
        return await self._request(f"nti/territory/Autonomous-region/{region_id}")

    async def get_country_spain(self) -> dict[str, Any]:
        return await self._request("nti/territory/Country/España")


# =============================================================================
# MCP SERVER
# =============================================================================

mcp = FastMCP("datos-gob-es")

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
            DatasetSummary.from_api_item(item, lang).model_dump(exclude_none=True)
            for item in items
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
    if isinstance(e, DatosGobClientError):
        return json.dumps({"error": e.message, "status_code": e.status_code}, ensure_ascii=False)
    return json.dumps({"error": str(e)}, ensure_ascii=False)


async def _fetch_all_pages(
    fetch_fn,
    max_results: int = 2000,
    sort: str | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Fetch all pages from an API endpoint up to max_results.

    Args:
        fetch_fn: The client method to call for fetching data.
        max_results: Maximum number of results to fetch.
        sort: Sort field for the query.
        **kwargs: Additional arguments to pass to fetch_fn.

    Returns:
        List of all items fetched across multiple pages.
    """
    all_items: list[dict[str, Any]] = []
    page = 0

    while len(all_items) < max_results:
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE, sort=sort)
        data = await fetch_fn(pagination=pagination, **kwargs)

        result = data.get("result", {})
        items = result.get("items", [])

        if not items:
            break

        all_items.extend(items)

        # Check if we got less than page_size (last page)
        if len(items) < DEFAULT_PAGE_SIZE:
            break

        page += 1

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
    items: list[dict[str, Any]],
    search_term: str,
    fields: list[str] = ["title", "description"]
) -> list[dict[str, Any]]:
    """Filter items to only include those with exact word matches."""
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
    format_filter: str | None = None,
    keyword: str | None = None,
) -> list[dict[str, Any]]:
    """Apply local filtering to dataset items after API query."""
    if not any([publisher, theme, format_filter, keyword]):
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

        # Check theme filter
        if theme:
            item_themes = item.get("theme", [])
            if isinstance(item_themes, str):
                item_themes = [item_themes]
            theme_match = any(theme.lower() in t.lower() for t in item_themes)
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

        # Check keyword filter
        if keyword:
            item_keywords = item.get("keyword", [])
            if isinstance(item_keywords, str):
                item_keywords = [item_keywords]
            kw_texts = []
            for kw in item_keywords:
                if isinstance(kw, dict):
                    kw_texts.append(kw.get("_value", "").lower())
                else:
                    kw_texts.append(str(kw).lower())
            if keyword.lower() not in " ".join(kw_texts):
                continue

        filtered.append(item)

    return filtered


# Fixed page size for all queries (API maximum)
DEFAULT_PAGE_SIZE = 200

# Maximum bytes to download for preview
PREVIEW_MAX_BYTES = 100 * 1024  # 100KB
PREVIEW_TIMEOUT = 10.0  # 10 seconds


def _normalize_format(format_str: str | None, media_type: str | None) -> str | None:
    """Normalize format string to a standard format identifier."""
    if format_str:
        fmt_lower = format_str.lower()
        if "csv" in fmt_lower or fmt_lower == "text/csv":
            return "csv"
        if "json" in fmt_lower or fmt_lower == "application/json":
            return "json"
    if media_type:
        mt_lower = media_type.lower()
        if "csv" in mt_lower:
            return "csv"
        if "json" in mt_lower:
            return "json"
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


def _parse_csv_preview(content: str, max_rows: int) -> DataPreview:
    """Parse CSV content and return a preview."""
    try:
        delimiter = _detect_csv_delimiter(content)
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows_list = list(reader)

        if not rows_list:
            return DataPreview(
                columns=[],
                rows=[],
                total_rows=0,
                format="csv",
                truncated=False,
            )

        columns = rows_list[0] if rows_list else []
        data_rows = rows_list[1 : max_rows + 1]
        total_rows = len(rows_list) - 1  # Exclude header

        return DataPreview(
            columns=columns,
            rows=data_rows,
            total_rows=total_rows,
            format="csv",
            truncated=len(rows_list) - 1 > max_rows,
        )
    except csv.Error as e:
        return DataPreview(
            columns=[],
            rows=[],
            format="csv",
            error=f"CSV parsing error: {e}",
        )


def _parse_json_preview(content: str, max_rows: int) -> DataPreview:
    """Parse JSON content and return a preview."""
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

        return DataPreview(
            columns=columns,
            rows=data_rows,
            total_rows=len(items),
            format="json",
            truncated=len(items) > max_rows,
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

    if normalized_format not in ("csv", "json"):
        return None  # Unsupported format

    try:
        async with httpx.AsyncClient(timeout=PREVIEW_TIMEOUT) as http_client:
            async with http_client.stream("GET", access_url) as response:
                response.raise_for_status()

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

        if normalized_format == "csv":
            return _parse_csv_preview(content, max_rows)
        elif normalized_format == "json":
            return _parse_json_preview(content, max_rows)

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


# =============================================================================
# DATASET TOOLS
# =============================================================================


@mcp.tool()
async def list_datasets(
    page: int = 0,
    sort: str | None = "-modified",
    lang: str | None = "es",
    fetch_all: bool = False,
    max_results: int = 2000,
) -> str:
    """List datasets from the Spanish open data catalog.

    Browse all available datasets with pagination. Use this to discover
    what public data is available from Spanish government institutions.

    Args:
        page: Page number (starting from 0). Ignored if fetch_all=True.
        sort: Sort field. Use '-' prefix for descending. Examples: '-modified', 'title', '-issued'.
        lang: Preferred language ('es', 'en', 'ca', 'eu', 'gl'). Default 'es'. Use None for all.
        fetch_all: If True, fetches all pages automatically up to max_results.
        max_results: Maximum results when fetch_all=True (default 2000, max 10000).

    Returns:
        JSON with datasets including title, description, publisher, and available formats.
    """
    try:
        max_results = min(max_results, 10000)  # Safety limit

        if fetch_all:
            all_items = await _fetch_all_pages(
                client.list_datasets, max_results=max_results, sort=sort
            )
            output = {
                "total_results": len(all_items),
                "fetch_all": True,
                "datasets": [
                    DatasetSummary.from_api_item(item, lang).model_dump(exclude_none=True)
                    for item in all_items
                ],
            }
            return json.dumps(output, ensure_ascii=False, indent=2)

        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE, sort=sort)
        data = await client.list_datasets(pagination)
        return _format_response(data, "dataset", lang)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_dataset(dataset_id: str, lang: str | None = "es") -> str:
    """Get detailed information about a specific dataset.

    Retrieve complete metadata for a dataset including all its distributions
    (downloadable files), description, publisher, and update frequency.

    Args:
        dataset_id: The dataset identifier (slug from the URL or URI).
        lang: Preferred language ('es', 'en', 'ca', 'eu', 'gl'). Default 'es'. Use None for all.

    Returns:
        JSON with full dataset details including download URLs.
    """
    try:
        data = await client.get_dataset(dataset_id)
        return _format_response(data, "dataset", lang)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def search_datasets(
    title: str | None = None,
    publisher: str | None = None,
    theme: str | None = None,
    format: str | None = None,
    keyword: str | None = None,
    spatial_type: str | None = None,
    spatial_value: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    exact_match: bool = False,
    page: int = 0,
    lang: str | None = "es",
    fetch_all: bool = False,
    max_results: int = 2000,
) -> str:
    """Search and filter datasets from the Spanish open data catalog.

    All filter parameters are optional and can be combined. When multiple filters
    are provided, the API query uses one filter and results are filtered locally.

    Args:
        title: Search text in dataset titles.
        publisher: Publisher ID (e.g., 'EA0010587' for INE). Use list_publishers to find IDs.
        theme: Theme ID (e.g., 'economia', 'salud', 'educacion'). Use list_themes to find IDs.
        format: Format ID (e.g., 'csv', 'json', 'xml').
        keyword: Keyword/tag to filter by (e.g., 'presupuesto', 'poblacion').
        spatial_type: Geographic type ('Autonomia', 'Provincia').
        spatial_value: Geographic value ('Madrid', 'Cataluna', 'Pais-Vasco').
        date_start: Start date 'YYYY-MM-DDTHH:mmZ' (e.g., '2024-01-01T00:00Z').
        date_end: End date 'YYYY-MM-DDTHH:mmZ' (e.g., '2024-12-31T23:59Z').
        exact_match: If True with title, match whole words only (e.g., 'DANA' won't match 'ciudadana').
        page: Page number (starting from 0). Ignored if fetch_all=True.
        lang: Preferred language ('es', 'en', 'ca', 'eu', 'gl'). Default 'es'. Use None for all.
        fetch_all: If True, fetches all pages automatically up to max_results.
        max_results: Maximum results when fetch_all=True (default 2000, max 10000).

    Returns:
        JSON with matching datasets.
    """
    try:
        max_results = min(max_results, 10000)  # Safety limit
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE)
        data: dict[str, Any] | None = None
        local_filters: dict[str, Any] = {}

        # Priority order for API query: title > date > spatial > publisher > theme > format > keyword
        if title:
            if exact_match:
                # Search multiple pages to find exact matches
                max_pages = 50 if fetch_all else 10
                all_filtered_items: list[dict[str, Any]] = []

                for current_page in range(max_pages):
                    page_pagination = PaginationParams(page=current_page, page_size=DEFAULT_PAGE_SIZE)
                    data = await client.search_datasets_by_title(title, page_pagination)

                    result = data.get("result", {})
                    items = result.get("items", [])

                    if not items:
                        break

                    filtered = _filter_items_by_exact_match(items, title)
                    # Apply local filters
                    filtered = _filter_datasets_locally(
                        filtered, publisher, theme, format, keyword
                    )
                    all_filtered_items.extend(filtered)

                    if not fetch_all and len(all_filtered_items) >= DEFAULT_PAGE_SIZE:
                        break
                    if fetch_all and len(all_filtered_items) >= max_results:
                        break

                if fetch_all:
                    all_filtered_items = all_filtered_items[:max_results]
                    output = {
                        "total_results": len(all_filtered_items),
                        "fetch_all": True,
                        "exact_match": True,
                        "datasets": [
                            DatasetSummary.from_api_item(item, lang).model_dump(exclude_none=True)
                            for item in all_filtered_items
                        ],
                    }
                else:
                    start_idx = page * DEFAULT_PAGE_SIZE
                    end_idx = start_idx + DEFAULT_PAGE_SIZE
                    paginated_items = all_filtered_items[start_idx:end_idx]

                    output = {
                        "total_in_page": len(paginated_items),
                        "total_exact_matches": len(all_filtered_items),
                        "page": page,
                        "items_per_page": DEFAULT_PAGE_SIZE,
                        "exact_match": True,
                        "datasets": [
                            DatasetSummary.from_api_item(item, lang).model_dump(exclude_none=True)
                            for item in paginated_items
                        ],
                    }
                return json.dumps(output, ensure_ascii=False, indent=2)
            else:
                data = await client.search_datasets_by_title(title, pagination)
                local_filters = {"publisher": publisher, "theme": theme, "format": format, "keyword": keyword}

        elif date_start and date_end:
            data = await client.get_datasets_by_date_range(date_start, date_end, pagination)
            local_filters = {"publisher": publisher, "theme": theme, "format": format, "keyword": keyword}

        elif spatial_type and spatial_value:
            data = await client.get_datasets_by_spatial(spatial_type, spatial_value, pagination)
            local_filters = {"publisher": publisher, "theme": theme, "format": format, "keyword": keyword}

        elif publisher:
            data = await client.get_datasets_by_publisher(publisher, pagination)
            local_filters = {"theme": theme, "format": format, "keyword": keyword}

        elif theme:
            data = await client.get_datasets_by_theme(theme, pagination)
            local_filters = {"publisher": publisher, "format": format, "keyword": keyword}

        elif format:
            data = await client.get_datasets_by_format(format, pagination)
            local_filters = {"publisher": publisher, "theme": theme, "keyword": keyword}

        elif keyword:
            data = await client.get_datasets_by_keyword(keyword, pagination)
            local_filters = {"publisher": publisher, "theme": theme, "format": format}

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
                local_filters.get("format"),
                local_filters.get("keyword"),
            )
            data["result"]["items"] = filtered_items

        return _format_response(data, "dataset", lang) if data else json.dumps({"error": "No data"})

    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def semantic_search(
    query: str,
    top_k: int = 20,
    min_score: float = 0.3,
    rebuild_index: bool = False,
) -> str:
    """Search datasets by meaning using AI embeddings.

    This tool understands natural language queries and finds semantically
    relevant datasets even if exact keywords don't match. For example,
    searching "unemployment data" will find datasets about "employment statistics".

    Note: First use may take 30-60 seconds to build the embedding index.
    Subsequent searches are fast (<1 second). The index is cached on disk.

    Args:
        query: Natural language search query (e.g., "unemployment statistics in Valencia").
        top_k: Maximum number of results to return (default 20, max 100).
        min_score: Minimum similarity score 0-1 (default 0.3). Higher = more relevant.
        rebuild_index: Force rebuild the embedding index (slow, use sparingly).

    Returns:
        JSON with matching datasets ranked by semantic relevance score.
    """
    if not EMBEDDINGS_AVAILABLE:
        return json.dumps({
            "error": "Semantic search not available. Install dependencies with: pip install sentence-transformers numpy",
            "suggestion": "Use search_datasets with title parameter for keyword-based search instead."
        }, ensure_ascii=False)

    try:
        top_k = min(top_k, 100)  # Safety limit

        # Check if we need to build or rebuild the index
        if rebuild_index:
            embedding_index.clear_cache()

        # Try to load from cache first
        if not embedding_index._initialized:
            cache_loaded = embedding_index._load_cache()
            if not cache_loaded:
                # Need to build index from scratch
                await embedding_index.build_index(client, max_datasets=5000)

        # Perform semantic search
        results = embedding_index.search(query, top_k=top_k, min_score=min_score)

        output = {
            "query": query,
            "total_results": len(results),
            "min_score": min_score,
            "datasets": results,
        }

        return json.dumps(output, ensure_ascii=False, indent=2)

    except Exception as e:
        return _handle_error(e)


# =============================================================================
# DISTRIBUTION TOOLS
# =============================================================================


@mcp.tool()
async def get_distributions(
    dataset_id: str | None = None,
    format: str | None = None,
    include_preview: bool = False,
    preview_rows: int = 10,
    page: int = 0,
) -> str:
    """Get downloadable files (distributions) from the open data catalog.

    Can filter by dataset or format. If no filters provided, lists all distributions.

    Args:
        dataset_id: Get distributions for a specific dataset.
        format: Filter by format (e.g., 'csv', 'json', 'xml').
        include_preview: Include data preview for CSV/JSON files (only with dataset_id).
        preview_rows: Number of preview rows (default 10, max 50).
        page: Page number (starting from 0).

    Returns:
        JSON with distributions including download URLs and formats.
    """
    try:
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE)

        if dataset_id:
            data = await client.get_distributions_by_dataset(dataset_id, pagination)
            if include_preview:
                preview_rows = min(max(1, preview_rows), 50)
                return await _format_response_with_preview(data, preview_rows)
        elif format:
            data = await client.get_distributions_by_format(format, pagination)
        else:
            data = await client.list_distributions(pagination)

        return _format_response(data, "distribution")
    except Exception as e:
        return _handle_error(e)


# =============================================================================
# METADATA TOOLS
# =============================================================================


@mcp.tool()
async def list_publishers(
    page: int = 0,
) -> str:
    """List all data publishers (government organizations).

    Get a list of all institutions that publish data on datos.gob.es.
    Use the publisher IDs to filter datasets by organization.

    Args:
        page: Page number (starting from 0).

    Returns:
        JSON with publisher organizations and their IDs.
    """
    try:
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_publishers(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def list_spatial_coverage(
    page: int = 0,
) -> str:
    """List all geographic coverage options.

    Get available geographic areas that datasets can cover.
    Includes autonomous regions, provinces, and municipalities.

    Args:
        page: Page number (starting from 0).

    Returns:
        JSON with geographic coverage options.
    """
    try:
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_spatial_coverage(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def list_themes(
    page: int = 0,
) -> str:
    """List all dataset categories/themes.

    Get all topic categories used to classify datasets.
    Common themes: economia, hacienda, educacion, salud, transporte, etc.

    Args:
        page: Page number (starting from 0).

    Returns:
        JSON with available themes and their labels.
    """
    try:
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_themes(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


# =============================================================================
# NTI TOOLS (Norma Técnica de Interoperabilidad)
# =============================================================================


@mcp.tool()
async def list_public_sectors(
    page: int = 0,
) -> str:
    """List all public sectors from NTI taxonomy.

    Get sectors defined by Spain's Technical Interoperability Standard.
    Includes: comercio, educacion, salud, justicia, etc.

    Args:
        page: Page number (starting from 0).

    Returns:
        JSON with public sector categories.
    """
    try:
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_public_sectors(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_public_sector(sector_id: str) -> str:
    """Get details about a specific public sector.

    Retrieve information about a sector from the NTI taxonomy.

    Args:
        sector_id: Sector identifier (e.g., 'comercio', 'educacion', 'salud').

    Returns:
        JSON with sector details.
    """
    try:
        data = await client.get_public_sector(sector_id)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def list_provinces(
    page: int = 0,
) -> str:
    """List all Spanish provinces.

    Get the 50 provinces of Spain plus Ceuta and Melilla.

    Args:
        page: Page number (starting from 0).

    Returns:
        JSON with Spanish provinces.
    """
    try:
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_provinces(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_province(province_id: str) -> str:
    """Get details about a specific Spanish province.

    Args:
        province_id: Province name (e.g., 'Madrid', 'Barcelona', 'Sevilla').

    Returns:
        JSON with province details.
    """
    try:
        data = await client.get_province(province_id)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def list_autonomous_regions(
    page: int = 0,
) -> str:
    """List all Spanish autonomous regions (Comunidades Autónomas).

    Get Spain's 17 autonomous communities plus Ceuta and Melilla.

    Args:
        page: Page number (starting from 0).

    Returns:
        JSON with autonomous regions.
    """
    try:
        pagination = PaginationParams(page=page, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_autonomous_regions(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_autonomous_region(region_id: str) -> str:
    """Get details about a specific autonomous region.

    Args:
        region_id: Region identifier (e.g., 'Comunidad-Madrid', 'Cataluna', 'Andalucia').

    Returns:
        JSON with region details.
    """
    try:
        data = await client.get_autonomous_region(region_id)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_country_spain() -> str:
    """Get information about Spain as a country.

    Retrieve Spain's country-level information from the NTI geographic taxonomy.

    Returns:
        JSON with Spain country data.
    """
    try:
        data = await client.get_country_spain()
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


# =============================================================================
# RESOURCES - Static Catalogs (cached data for quick access)
# =============================================================================


@mcp.resource("catalog://themes")
async def resource_themes() -> str:
    """
    Catálogo completo de temáticas/categorías disponibles en datos.gob.es.

    Este recurso proporciona acceso rápido a todas las categorías temáticas
    que se usan para clasificar los datasets del portal de datos abiertos.

    Temáticas principales incluyen:
    - economia: Datos económicos, PIB, comercio, empresas
    - hacienda: Presupuestos, impuestos, gasto público
    - educacion: Sistema educativo, universidades, becas
    - salud: Sanidad, hospitales, epidemiología
    - medio-ambiente: Calidad del aire, agua, residuos
    - transporte: Movilidad, carreteras, transporte público
    - turismo: Visitantes, alojamientos, destinos
    - empleo: Mercado laboral, paro, contratación
    - sector-publico: Administración, funcionarios
    - ciencia-tecnologia: I+D, innovación, patentes
    """
    try:
        pagination = PaginationParams(page=0, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_themes(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.resource("catalog://publishers")
async def resource_publishers() -> str:
    """
    Catálogo de organismos publicadores de datos en datos.gob.es.

    Este recurso lista todas las instituciones gubernamentales y organismos
    públicos que publican datos abiertos en el portal.

    Incluye organismos como:
    - INE (Instituto Nacional de Estadística)
    - Ministerios del Gobierno de España
    - Comunidades Autónomas
    - Ayuntamientos y Diputaciones
    - Organismos autónomos y agencias estatales
    - Universidades públicas

    Cada publicador tiene un ID único que puede usarse para filtrar datasets.
    """
    try:
        pagination = PaginationParams(page=0, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_publishers(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.resource("catalog://provinces")
async def resource_provinces() -> str:
    """
    Catálogo de las 52 provincias españolas según la NTI.

    Este recurso proporciona la lista completa de provincias de España,
    incluyendo las 50 provincias peninsulares e insulares más Ceuta y Melilla.

    Organización territorial:
    - 50 provincias distribuidas en 17 Comunidades Autónomas
    - 2 ciudades autónomas: Ceuta y Melilla

    Cada provincia tiene un identificador que puede usarse para filtrar
    datasets por cobertura geográfica provincial.

    Ejemplos de IDs: Madrid, Barcelona, Sevilla, Valencia, Vizcaya, etc.
    """
    try:
        pagination = PaginationParams(page=0, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_provinces(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.resource("catalog://autonomous-regions")
async def resource_autonomous_regions() -> str:
    """
    Catálogo de las 17 Comunidades Autónomas y 2 Ciudades Autónomas de España.

    Este recurso proporciona información sobre la organización territorial
    de España a nivel autonómico según la Norma Técnica de Interoperabilidad.

    Comunidades Autónomas:
    - Andalucía, Aragón, Asturias, Baleares, Canarias
    - Cantabria, Castilla-La Mancha, Castilla y León
    - Cataluña, Comunidad Valenciana, Extremadura
    - Galicia, Madrid, Murcia, Navarra, País Vasco, La Rioja

    Ciudades Autónomas:
    - Ceuta, Melilla

    Ejemplos de IDs: Comunidad-Madrid, Cataluna, Andalucia, Pais-Vasco
    """
    try:
        pagination = PaginationParams(page=0, page_size=DEFAULT_PAGE_SIZE)
        data = await client.list_autonomous_regions(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


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
        data = await client.get_datasets_by_keyword(keyword, pagination)
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


# =============================================================================
# NOTIFICATIONS - Webhooks and Dataset Watching
# =============================================================================

# Import and register webhook/notification tools
try:
    from notifications.webhook import register_webhook_tools
    register_webhook_tools(mcp)
except ImportError:
    # Notifications not available
    pass


# Export mcp for FastMCP Cloud
# The 'mcp' object is the FastMCP server instance that FastMCP Cloud will use

if __name__ == "__main__":
    mcp.run()
