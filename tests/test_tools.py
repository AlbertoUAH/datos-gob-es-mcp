"""Tests for MCP tools."""

import json
import pytest
from server import (
    get_dataset,
    search_datasets,
    get_distributions,
    list_publishers,
    list_themes,
    list_public_sectors,
    list_provinces,
    list_autonomous_regions,
)


def get_tool_fn(tool):
    """Extract the underlying async function from a FastMCP tool."""
    if hasattr(tool, 'fn'):
        return tool.fn
    return tool


class TestDatasetTools:
    """Tests for dataset-related MCP tools."""

    @pytest.mark.asyncio
    async def test_get_dataset(self, mock_api):
        """Test get_dataset tool."""
        fn = get_tool_fn(get_dataset)
        result = await fn("test-dataset-123")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_datasets_by_title(self, mock_api):
        """Test search_datasets with title filter."""
        fn = get_tool_fn(search_datasets)
        result = await fn(title="empleo")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_datasets_by_publisher(self, mock_api):
        """Test search_datasets with publisher filter."""
        fn = get_tool_fn(search_datasets)
        result = await fn(publisher="E00003901")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_datasets_by_theme(self, mock_api):
        """Test search_datasets with theme filter."""
        fn = get_tool_fn(search_datasets)
        result = await fn(theme="economia")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_datasets_by_format(self, mock_api):
        """Test search_datasets with format filter."""
        fn = get_tool_fn(search_datasets)
        result = await fn(format="csv")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_datasets_by_keyword(self, mock_api):
        """Test search_datasets with keyword filter."""
        fn = get_tool_fn(search_datasets)
        result = await fn(keyword="presupuesto")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_datasets_by_spatial(self, mock_api):
        """Test search_datasets with spatial filter."""
        fn = get_tool_fn(search_datasets)
        result = await fn(spatial_type="Autonomia", spatial_value="Comunidad-Madrid")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_datasets_by_date_range(self, mock_api):
        """Test search_datasets with date range filter."""
        fn = get_tool_fn(search_datasets)
        result = await fn(
            date_start="2024-01-01T00:00Z",
            date_end="2024-12-31T23:59Z"
        )
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_datasets_no_filter(self, mock_api):
        """Test search_datasets with no filters (lists all)."""
        fn = get_tool_fn(search_datasets)
        result = await fn()
        data = json.loads(result)

        assert "datasets" in data


class TestDistributionTools:
    """Tests for distribution-related MCP tools."""

    @pytest.mark.asyncio
    async def test_get_distributions_all(self, mock_api):
        """Test get_distributions with no filter (lists all)."""
        fn = get_tool_fn(get_distributions)
        result = await fn()
        data = json.loads(result)

        assert "distributions" in data
        assert "total_in_page" in data

    @pytest.mark.asyncio
    async def test_get_distributions_by_dataset(self, mock_api):
        """Test get_distributions with dataset_id filter."""
        fn = get_tool_fn(get_distributions)
        result = await fn(dataset_id="test-dataset-123")
        data = json.loads(result)

        assert "distributions" in data

    @pytest.mark.asyncio
    async def test_get_distributions_by_format(self, mock_api):
        """Test get_distributions with format filter."""
        fn = get_tool_fn(get_distributions)
        result = await fn(format="csv")
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
        # Response format varies based on cache: "total_items" (cached) or "total_in_page" (API)
        assert "total_items" in data or "total_in_page" in data

    @pytest.mark.asyncio
    async def test_list_themes(self, mock_api):
        """Test list_themes tool."""
        fn = get_tool_fn(list_themes)
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
    async def test_list_provinces(self, mock_api):
        """Test list_provinces tool."""
        fn = get_tool_fn(list_provinces)
        result = await fn()
        data = json.loads(result)

        assert "items" in data

    @pytest.mark.asyncio
    async def test_list_autonomous_regions(self, mock_api):
        """Test list_autonomous_regions tool."""
        fn = get_tool_fn(list_autonomous_regions)
        result = await fn()
        data = json.loads(result)

        assert "items" in data


class TestToolErrorHandling:
    """Tests for error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_tool_returns_error_json(self):
        """Test that tools return error as JSON when API fails."""
        import respx

        # Use list_publishers with use_cache=False to force API call
        fn = get_tool_fn(list_publishers)

        with respx.mock:
            respx.get(url__regex=r".*").respond(status_code=500, json={"error": "Server error"})

            result = await fn(use_cache=False)
            data = json.loads(result)

            assert "error" in data
            assert data["status_code"] == 500
