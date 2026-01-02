"""Tests for MCP resources."""

import json
import pytest
from server import (
    resource_dataset,
    resource_theme_datasets,
    resource_publisher_datasets,
    resource_format_datasets,
    resource_keyword_datasets,
)


def get_resource_fn(resource):
    """Extract the underlying async function from a FastMCP resource."""
    if hasattr(resource, 'fn'):
        return resource.fn
    return resource


class TestDynamicResources:
    """Tests for dynamic MCP resource templates."""

    @pytest.mark.asyncio
    async def test_resource_dataset(self, mock_api):
        """Test dataset://{dataset_id} resource template."""
        fn = get_resource_fn(resource_dataset)
        result = await fn("test-dataset-123")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_resource_theme_datasets(self, mock_api):
        """Test theme://{theme_id} resource template."""
        fn = get_resource_fn(resource_theme_datasets)
        result = await fn("economia")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_resource_publisher_datasets(self, mock_api):
        """Test publisher://{publisher_id} resource template."""
        fn = get_resource_fn(resource_publisher_datasets)
        result = await fn("E00003901")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_resource_format_datasets(self, mock_api):
        """Test format://{format_id} resource template."""
        fn = get_resource_fn(resource_format_datasets)
        result = await fn("csv")
        data = json.loads(result)

        assert "datasets" in data

    @pytest.mark.asyncio
    async def test_resource_keyword_datasets(self, mock_api):
        """Test keyword://{keyword} resource template."""
        fn = get_resource_fn(resource_keyword_datasets)
        result = await fn("presupuesto")
        data = json.loads(result)

        assert "datasets" in data


class TestResourceErrorHandling:
    """Tests for error handling in MCP resources."""

    @pytest.mark.asyncio
    async def test_resource_returns_error_json(self):
        """Test that resources return error as JSON when API fails."""
        import respx

        fn = get_resource_fn(resource_dataset)

        with respx.mock:
            respx.get(url__regex=r".*").respond(status_code=500, json={"error": "Server error"})

            result = await fn("test-dataset-123")
            data = json.loads(result)

            assert "error" in data
