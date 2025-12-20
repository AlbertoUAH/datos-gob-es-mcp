"""Pydantic models for datos.gob.es API responses."""

from typing import Any

from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    """Parameters for paginated API requests."""

    page: int = Field(default=0, ge=0, description="Page number (0-indexed)")
    page_size: int = Field(default=10, ge=1, le=50, description="Items per page (max 50)")
    sort: str | None = Field(
        default=None, description="Sort field(s), prefix with - for descending"
    )


class MultilingualText(BaseModel):
    """Multilingual text field from the API."""

    value: str = Field(alias="_value")
    lang: str = Field(alias="_lang")

    class Config:
        populate_by_name = True


class ApiPagination(BaseModel):
    """Pagination info from API response."""

    page: int = 0
    items_per_page: int = Field(default=10, alias="itemsPerPage")
    start_index: int = Field(default=0, alias="startIndex")
    first: str | None = None
    next: str | None = None
    prev: str | None = None

    class Config:
        populate_by_name = True


class ApiResult(BaseModel):
    """Generic API result wrapper."""

    about: str = Field(alias="_about")
    items: list[dict[str, Any]] = Field(default_factory=list)
    page: int = 0
    items_per_page: int = Field(default=10, alias="itemsPerPage")
    start_index: int = Field(default=0, alias="startIndex")

    class Config:
        populate_by_name = True
        extra = "allow"


class ApiResponse(BaseModel):
    """Top-level API response structure."""

    format: str = "linked-data-api"
    version: str = "0.2"
    result: ApiResult

    class Config:
        extra = "allow"


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
    def from_api_item(cls, item: dict[str, Any]) -> "DatasetSummary":
        """Create a DatasetSummary from an API response item."""
        title = cls._extract_text(item.get("title"))
        description = cls._extract_text(item.get("description"))

        distributions = item.get("distribution", [])
        if isinstance(distributions, dict):
            distributions = [distributions]

        # Handle theme which can be string or list
        theme = item.get("theme")
        if isinstance(theme, str):
            theme = [theme]
        elif isinstance(theme, list):
            theme = [t if isinstance(t, str) else str(t) for t in theme]

        # Handle keywords which can contain multilingual objects
        keywords = cls._extract_keywords(item.get("keyword", []))

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
    def _extract_keywords(value: Any) -> list[str] | None:
        """Extract keywords handling multilingual format."""
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
                    keywords.append(item.get("_value", str(item)))
            return keywords if keywords else None
        return None

    @staticmethod
    def _extract_text(value: Any) -> str | list[str] | None:
        """Extract text from multilingual field."""
        if value is None:
            return None
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
            return texts if len(texts) > 1 else (texts[0] if texts else None)
        return str(value)


class DistributionSummary(BaseModel):
    """Simplified distribution representation."""

    uri: str
    title: str | None = None
    access_url: str | None = None
    format: str | None = None
    media_type: str | None = None

    @classmethod
    def from_api_item(cls, item: dict[str, Any]) -> "DistributionSummary":
        """Create a DistributionSummary from an API response item."""
        title = item.get("title")
        if isinstance(title, dict):
            title = title.get("_value")
        elif isinstance(title, list) and title:
            title = title[0].get("_value") if isinstance(title[0], dict) else title[0]

        return cls(
            uri=item.get("_about", ""),
            title=title,
            access_url=item.get("accessURL"),
            format=item.get("format"),
            media_type=item.get("mediaType"),
        )
