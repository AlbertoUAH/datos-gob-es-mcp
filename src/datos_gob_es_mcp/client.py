"""HTTP client for datos.gob.es API."""

from typing import Any
from urllib.parse import urljoin

import httpx

from .models import PaginationParams


class DatosGobClientError(Exception):
    """Exception raised for API client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DatosGobClient:
    """Async HTTP client for the datos.gob.es API."""

    BASE_URL = "https://datos.gob.es/apidata/"
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        """Initialize the client.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout

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
        """Make an async HTTP request to the API.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            JSON response as dictionary.

        Raises:
            DatosGobClientError: If the request fails.
        """
        url = urljoin(self.BASE_URL, endpoint)
        if not url.endswith(".json"):
            url = f"{url}.json"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as e:
                raise DatosGobClientError(f"Request timed out: {e}") from e
            except httpx.HTTPStatusError as e:
                raise DatosGobClientError(
                    f"HTTP error {e.response.status_code}: {e.response.text}",
                    status_code=e.response.status_code,
                ) from e
            except httpx.RequestError as e:
                raise DatosGobClientError(f"Request failed: {e}") from e

    # =========================================================================
    # DATASETS
    # =========================================================================

    async def list_datasets(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        """Get all datasets with pagination.

        Args:
            pagination: Pagination parameters.

        Returns:
            API response with datasets.
        """
        params = self._build_params(pagination)
        return await self._request("catalog/dataset", params)

    async def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Get a specific dataset by ID.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            API response with the dataset.
        """
        return await self._request(f"catalog/dataset/{dataset_id}")

    async def search_datasets_by_title(
        self, title: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Search datasets by title (partial match).

        Args:
            title: Title text to search for.
            pagination: Pagination parameters.

        Returns:
            API response with matching datasets.
        """
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/title/{title}", params)

    async def get_datasets_by_publisher(
        self, publisher_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get datasets by publisher ID.

        Args:
            publisher_id: Publisher identifier (e.g., 'A16003011').
            pagination: Pagination parameters.

        Returns:
            API response with datasets.
        """
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/publisher/{publisher_id}", params)

    async def get_datasets_by_theme(
        self, theme_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get datasets by theme/category.

        Args:
            theme_id: Theme identifier (e.g., 'hacienda', 'economia').
            pagination: Pagination parameters.

        Returns:
            API response with datasets.
        """
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/theme/{theme_id}", params)

    async def get_datasets_by_format(
        self, format_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get datasets by distribution format.

        Args:
            format_id: Format identifier (e.g., 'csv', 'json', 'xml').
            pagination: Pagination parameters.

        Returns:
            API response with datasets.
        """
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/format/{format_id}", params)

    async def get_datasets_by_keyword(
        self, keyword: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get datasets by keyword/tag.

        Args:
            keyword: Keyword to search for (e.g., 'gastos', 'presupuesto').
            pagination: Pagination parameters.

        Returns:
            API response with datasets.
        """
        params = self._build_params(pagination)
        return await self._request(f"catalog/dataset/keyword/{keyword}", params)

    async def get_datasets_by_spatial(
        self,
        spatial_word1: str,
        spatial_word2: str,
        pagination: PaginationParams | None = None,
    ) -> dict[str, Any]:
        """Get datasets by geographic scope.

        Args:
            spatial_word1: First spatial identifier (e.g., 'Autonomia').
            spatial_word2: Second spatial identifier (e.g., 'Pais-Vasco').
            pagination: Pagination parameters.

        Returns:
            API response with datasets.
        """
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
        """Get datasets modified within a date range.

        Args:
            begin_date: Start date in format 'YYYY-MM-DDTHH:mmZ'.
            end_date: End date in format 'YYYY-MM-DDTHH:mmZ'.
            pagination: Pagination parameters.

        Returns:
            API response with datasets.
        """
        params = self._build_params(pagination)
        return await self._request(
            f"catalog/dataset/modified/begin/{begin_date}/end/{end_date}", params
        )

    # =========================================================================
    # DISTRIBUTIONS
    # =========================================================================

    async def list_distributions(
        self, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get all distributions with pagination.

        Args:
            pagination: Pagination parameters.

        Returns:
            API response with distributions.
        """
        params = self._build_params(pagination)
        return await self._request("catalog/distribution", params)

    async def get_distributions_by_dataset(
        self, dataset_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get distributions for a specific dataset.

        Args:
            dataset_id: Dataset identifier.
            pagination: Pagination parameters.

        Returns:
            API response with distributions.
        """
        params = self._build_params(pagination)
        return await self._request(f"catalog/distribution/dataset/{dataset_id}", params)

    async def get_distributions_by_format(
        self, format_id: str, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get distributions by format.

        Args:
            format_id: Format identifier (e.g., 'csv', 'json').
            pagination: Pagination parameters.

        Returns:
            API response with distributions.
        """
        params = self._build_params(pagination)
        return await self._request(f"catalog/distribution/format/{format_id}", params)

    # =========================================================================
    # METADATA (Publishers, Spatial, Themes)
    # =========================================================================

    async def list_publishers(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        """Get all data publishers.

        Args:
            pagination: Pagination parameters.

        Returns:
            API response with publishers.
        """
        params = self._build_params(pagination)
        return await self._request("catalog/publisher", params)

    async def list_spatial_coverage(
        self, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get all geographic coverage options.

        Args:
            pagination: Pagination parameters.

        Returns:
            API response with spatial coverage options.
        """
        params = self._build_params(pagination)
        return await self._request("catalog/spatial", params)

    async def list_themes(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        """Get all dataset themes/categories.

        Args:
            pagination: Pagination parameters.

        Returns:
            API response with themes.
        """
        params = self._build_params(pagination)
        return await self._request("catalog/theme", params)

    # =========================================================================
    # NTI (Norma Técnica de Interoperabilidad)
    # =========================================================================

    async def list_public_sectors(
        self, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get all public sectors from NTI taxonomy.

        Args:
            pagination: Pagination parameters.

        Returns:
            API response with public sectors.
        """
        params = self._build_params(pagination)
        return await self._request("nti/public-sector", params)

    async def get_public_sector(self, sector_id: str) -> dict[str, Any]:
        """Get a specific public sector by ID.

        Args:
            sector_id: Sector identifier (e.g., 'comercio', 'educacion').

        Returns:
            API response with the sector.
        """
        return await self._request(f"nti/public-sector/{sector_id}")

    async def list_provinces(self, pagination: PaginationParams | None = None) -> dict[str, Any]:
        """Get all Spanish provinces.

        Args:
            pagination: Pagination parameters.

        Returns:
            API response with provinces.
        """
        params = self._build_params(pagination)
        return await self._request("nti/territory/Province", params)

    async def get_province(self, province_id: str) -> dict[str, Any]:
        """Get a specific province by ID.

        Args:
            province_id: Province identifier (e.g., 'Madrid', 'Barcelona').

        Returns:
            API response with the province.
        """
        return await self._request(f"nti/territory/Province/{province_id}")

    async def list_autonomous_regions(
        self, pagination: PaginationParams | None = None
    ) -> dict[str, Any]:
        """Get all Spanish autonomous regions (Comunidades Autónomas).

        Args:
            pagination: Pagination parameters.

        Returns:
            API response with autonomous regions.
        """
        params = self._build_params(pagination)
        return await self._request("nti/territory/Autonomous-region", params)

    async def get_autonomous_region(self, region_id: str) -> dict[str, Any]:
        """Get a specific autonomous region by ID.

        Args:
            region_id: Region identifier (e.g., 'Comunidad-Madrid', 'Cataluna').

        Returns:
            API response with the region.
        """
        return await self._request(f"nti/territory/Autonomous-region/{region_id}")

    async def get_country_spain(self) -> dict[str, Any]:
        """Get Spain country information.

        Returns:
            API response with Spain data.
        """
        return await self._request("nti/territory/Country/España")
