"""Tests for MCP tools."""

import json
import pytest
from server import (
    list_datasets,
    get_dataset,
    search_datasets_by_title,
    get_datasets_by_publisher,
    get_datasets_by_theme,
    get_datasets_by_format,
    get_datasets_by_keyword,
    get_datasets_by_spatial,
    get_datasets_by_date_range,
    list_distributions,
    get_distributions_by_dataset,
    get_distributions_by_format,
    list_publishers,
    list_themes,
    list_spatial_coverage,
    list_public_sectors,
    get_public_sector,
    list_provinces,
    get_province,
    list_autonomous_regions,
    get_autonomous_region,
    get_country_spain,
)


def get_tool_fn(tool):
    """Extract the underlying async function from a FastMCP tool."""
    if hasattr(tool, 'fn'):
        return tool.fn
    return tool


class TestDatasetTools:
    """Tests for dataset-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_datasets(self, mock_api):
        """Test list_datasets tool."""
        fn = get_tool_fn(list_datasets)
        result = await fn()
        data = json.loads(result)

        assert "datasets" in data
        assert "total_in_page" in data
        assert "page" in data

    @pytest.mark.asyncio
    async def test_list_datasets_with_pagination(self, mock_api):
        """Test list_datasets with custom pagination."""
        fn = get_tool_fn(list_datasets)
        result = await fn(page=1, sort="-issued")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_get_dataset(self, mock_api):
        """Test get_dataset tool."""
        fn = get_tool_fn(get_dataset)
        result = await fn("test-dataset-123")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_datasets_by_title(self, mock_api):
        """Test search_datasets_by_title tool."""
        fn = get_tool_fn(search_datasets_by_title)
        result = await fn("empleo")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_get_datasets_by_publisher(self, mock_api):
        """Test get_datasets_by_publisher tool."""
        fn = get_tool_fn(get_datasets_by_publisher)
        result = await fn("E00003901")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_get_datasets_by_theme(self, mock_api):
        """Test get_datasets_by_theme tool."""
        fn = get_tool_fn(get_datasets_by_theme)
        result = await fn("economia")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_get_datasets_by_format(self, mock_api):
        """Test get_datasets_by_format tool."""
        fn = get_tool_fn(get_datasets_by_format)
        result = await fn("csv")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_get_datasets_by_keyword(self, mock_api):
        """Test get_datasets_by_keyword tool."""
        fn = get_tool_fn(get_datasets_by_keyword)
        result = await fn("presupuesto")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_get_datasets_by_spatial(self, mock_api):
        """Test get_datasets_by_spatial tool."""
        fn = get_tool_fn(get_datasets_by_spatial)
        result = await fn("Autonomia", "Comunidad-Madrid")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_get_datasets_by_date_range(self, mock_api):
        """Test get_datasets_by_date_range tool."""
        fn = get_tool_fn(get_datasets_by_date_range)
        result = await fn(
            "2024-01-01T00:00Z",
            "2024-12-31T23:59Z"
        )
        data = json.loads(result)

        assert "datasets" in data


class TestDistributionTools:
    """Tests for distribution-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_distributions(self, mock_api):
        """Test list_distributions tool."""
        fn = get_tool_fn(list_distributions)
        result = await fn()
        data = json.loads(result)

        assert "distributions" in data
        assert "total_in_page" in data

    @pytest.mark.asyncio
    async def test_get_distributions_by_dataset(self, mock_api):
        """Test get_distributions_by_dataset tool."""
        fn = get_tool_fn(get_distributions_by_dataset)
        result = await fn("test-dataset-123")
        data = json.loads(result)

        assert "distributions" in data

    @pytest.mark.asyncio
    async def test_get_distributions_by_format(self, mock_api):
        """Test get_distributions_by_format tool."""
        fn = get_tool_fn(get_distributions_by_format)
        result = await fn("csv")
        data = json.loads(result)

        assert "distributions" in data


class TestMetadataTools:
    """Tests for metadata-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_publishers(self, mock_api):
        """Test list_publishers tool."""
        fn = get_tool_fn(list_publishers)
        result = await fn()
        data = json.loads(result)

        assert "items" in data
        assert "total_in_page" in data

    @pytest.mark.asyncio
    async def test_list_themes(self, mock_api):
        """Test list_themes tool."""
        fn = get_tool_fn(list_themes)
        result = await fn()
        data = json.loads(result)

        assert "items" in data

    @pytest.mark.asyncio
    async def test_list_spatial_coverage(self, mock_api):
        """Test list_spatial_coverage tool."""
        fn = get_tool_fn(list_spatial_coverage)
        result = await fn()
        data = json.loads(result)

        assert "items" in data


class TestNTITools:
    """Tests for NTI-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_public_sectors(self, mock_api):
        """Test list_public_sectors tool."""
        fn = get_tool_fn(list_public_sectors)
        result = await fn()
        data = json.loads(result)

        assert "items" in data

    @pytest.mark.asyncio
    async def test_get_public_sector(self, mock_api):
        """Test get_public_sector tool."""
        fn = get_tool_fn(get_public_sector)
        result = await fn("comercio")
        data = json.loads(result)

        assert "items" in data

    @pytest.mark.asyncio
    async def test_list_provinces(self, mock_api):
        """Test list_provinces tool."""
        fn = get_tool_fn(list_provinces)
        result = await fn()
        data = json.loads(result)

        assert "items" in data

    @pytest.mark.asyncio
    async def test_get_province(self, mock_api):
        """Test get_province tool."""
        fn = get_tool_fn(get_province)
        result = await fn("Madrid")
        data = json.loads(result)

        assert "items" in data

    @pytest.mark.asyncio
    async def test_list_autonomous_regions(self, mock_api):
        """Test list_autonomous_regions tool."""
        fn = get_tool_fn(list_autonomous_regions)
        result = await fn()
        data = json.loads(result)

        assert "items" in data

    @pytest.mark.asyncio
    async def test_get_autonomous_region(self, mock_api):
        """Test get_autonomous_region tool."""
        fn = get_tool_fn(get_autonomous_region)
        result = await fn("Comunidad-Madrid")
        data = json.loads(result)

        assert "items" in data

    @pytest.mark.asyncio
    async def test_get_country_spain(self, mock_api):
        """Test get_country_spain tool."""
        fn = get_tool_fn(get_country_spain)
        result = await fn()
        data = json.loads(result)

        assert "items" in data


class TestToolErrorHandling:
    """Tests for error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_tool_returns_error_json(self):
        """Test that tools return error as JSON when API fails."""
        import respx

        fn = get_tool_fn(list_datasets)

        with respx.mock:
            respx.get(url__regex=r".*").respond(status_code=500, json={"error": "Server error"})

            result = await fn()
            data = json.loads(result)

            assert "error" in data
            assert data["status_code"] == 500
