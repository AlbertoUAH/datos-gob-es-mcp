"""Tests for MCP tools."""

import json
import pytest
from server import (
    get_dataset,
    search_datasets,
    list_metadata,
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


class TestMetadataTools:
    """Tests for metadata-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_metadata_publishers(self, mock_api):
        """Test list_metadata tool with publishers type."""
        fn = get_tool_fn(list_metadata)
        result = await fn(metadata_type="publishers")
        data = json.loads(result)

        assert "items" in data
        assert "metadata_type" in data
        assert data["metadata_type"] == "publishers"

    @pytest.mark.asyncio
    async def test_list_metadata_themes(self, mock_api):
        """Test list_metadata tool with themes type."""
        fn = get_tool_fn(list_metadata)
        result = await fn(metadata_type="themes")
        data = json.loads(result)

        assert "items" in data
        assert "metadata_type" in data
        assert data["metadata_type"] == "themes"

    @pytest.mark.asyncio
    async def test_list_metadata_public_sectors(self, mock_api):
        """Test list_metadata tool with public_sectors type."""
        fn = get_tool_fn(list_metadata)
        result = await fn(metadata_type="public_sectors")
        data = json.loads(result)

        assert "items" in data
        assert "metadata_type" in data
        assert data["metadata_type"] == "public_sectors"

    @pytest.mark.asyncio
    async def test_list_metadata_provinces(self, mock_api):
        """Test list_metadata tool with provinces type."""
        fn = get_tool_fn(list_metadata)
        result = await fn(metadata_type="provinces")
        data = json.loads(result)

        assert "items" in data
        assert "metadata_type" in data
        assert data["metadata_type"] == "provinces"

    @pytest.mark.asyncio
    async def test_list_metadata_autonomous_regions(self, mock_api):
        """Test list_metadata tool with autonomous_regions type."""
        fn = get_tool_fn(list_metadata)
        result = await fn(metadata_type="autonomous_regions")
        data = json.loads(result)

        assert "items" in data
        assert "metadata_type" in data
        assert data["metadata_type"] == "autonomous_regions"

    @pytest.mark.asyncio
    async def test_list_metadata_invalid_type(self, mock_api):
        """Test list_metadata tool with invalid type returns error."""
        fn = get_tool_fn(list_metadata)
        result = await fn(metadata_type="invalid_type")
        data = json.loads(result)

        assert "error" in data
        assert "valid_types" in data


class TestToolErrorHandling:
    """Tests for error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_tool_returns_error_json(self):
        """Test that tools return error as JSON when API fails."""
        import respx

        # Use list_metadata with use_cache=False to force API call
        fn = get_tool_fn(list_metadata)

        with respx.mock:
            respx.get(url__regex=r".*").respond(status_code=500, json={"error": "Server error"})

            result = await fn(metadata_type="publishers", use_cache=False)
            data = json.loads(result)

            assert "error" in data
            assert data["status_code"] == 500
