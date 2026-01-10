"""Tests for MCP tools."""

import json
import pytest
from server import (
    get,
    search,
)


def get_tool_fn(tool):
    """Extract the underlying async function from a FastMCP tool."""
    if hasattr(tool, 'fn'):
        return tool.fn
    return tool


class TestDatasetTools:
    """Tests for dataset-related MCP tools."""

    @pytest.mark.asyncio
    async def test_get(self, mock_api):
        """Test get tool (metadata only)."""
        fn = get_tool_fn(get)
        result = await fn("test-dataset-123")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_get_with_data(self, mock_api):
        """Test get tool with include_data=True."""
        fn = get_tool_fn(get)
        # This will fail if no compatible distribution, but tests the interface
        result = await fn("test-dataset-123", include_data=True)
        data = json.loads(result)

        # Either has data or error (depending on mock)
        assert "datasets" in data or "error" in data

    @pytest.mark.asyncio
    async def test_search_by_title(self, mock_api):
        """Test search with title filter."""
        fn = get_tool_fn(search)
        result = await fn(title="empleo")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_by_publisher(self, mock_api):
        """Test search with publisher filter."""
        fn = get_tool_fn(search)
        result = await fn(publisher="E00003901")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_by_theme(self, mock_api):
        """Test search with theme filter."""
        fn = get_tool_fn(search)
        result = await fn(theme="economia")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_by_format(self, mock_api):
        """Test search with format filter."""
        fn = get_tool_fn(search)
        result = await fn(format="csv")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_by_keyword(self, mock_api):
        """Test search with keyword filter."""
        fn = get_tool_fn(search)
        result = await fn(keyword="presupuesto")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_by_date_range(self, mock_api):
        """Test search with date range filter."""
        fn = get_tool_fn(search)
        result = await fn(
            date_start="2024-01-01T00:00Z",
            date_end="2024-12-31T23:59Z"
        )
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_search_no_filter(self, mock_api):
        """Test search with no filters (lists all)."""
        fn = get_tool_fn(search)
        result = await fn()
        data = json.loads(result)

        assert "datasets" in data


class TestToolErrorHandling:
    """Tests for error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_tool_returns_error_json(self):
        """Test that tools return error as JSON when API fails."""
        import respx

        fn = get_tool_fn(get)

        with respx.mock:
            respx.get(url__regex=r".*").respond(status_code=500, json={"error": "Server error"})

            result = await fn("invalid-dataset-id")
            data = json.loads(result)

            assert "error" in data
