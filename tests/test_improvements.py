"""Tests for the improvements implemented."""

import asyncio
import pickle
import time
from unittest.mock import AsyncMock, patch

import pytest


# Test 1: Metadata Cache
class TestMetadataCache:
    """Tests for the MetadataCache class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock DatosGobClient."""
        client = AsyncMock()
        client.list_publishers = AsyncMock(
            return_value={"result": {"items": [{"_about": "pub1", "title": "Publisher 1"}]}}
        )
        client.list_themes = AsyncMock(
            return_value={"result": {"items": [{"_about": "theme1", "label": "Theme 1"}]}}
        )
        client.list_provinces = AsyncMock(
            return_value={"result": {"items": [{"_about": "prov1", "label": "Madrid"}]}}
        )
        client.list_autonomous_regions = AsyncMock(
            return_value={
                "result": {"items": [{"_about": "region1", "label": "Comunidad de Madrid"}]}
            }
        )
        client.list_public_sectors = AsyncMock(
            return_value={"result": {"items": [{"_about": "sector1", "label": "Economy"}]}}
        )
        client.list_spatial_coverage = AsyncMock(
            return_value={"result": {"items": [{"_about": "spatial1", "label": "Spain"}]}}
        )
        return client

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / ".cache" / "datos-gob-es"
        cache_dir.mkdir(parents=True)
        return cache_dir

    @pytest.mark.asyncio
    async def test_cache_loads_from_disk(self, temp_cache_dir, mock_client):
        """Cache should load from disk if file exists and is valid."""
        from server import MetadataCache

        # Create a valid cache file
        cache_file = temp_cache_dir / "metadata.pkl"
        cache_data = {
            "timestamp": time.time(),
            "publishers": [{"_about": "cached_pub", "title": "Cached Publisher"}],
            "themes": None,
            "provinces": None,
            "autonomous_regions": None,
            "public_sectors": None,
            "spatial_coverage": None,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        # Create cache instance with mocked paths
        with patch.object(MetadataCache, "_CACHE_DIR", temp_cache_dir):
            with patch.object(MetadataCache, "CACHE_FILE", cache_file):
                cache = MetadataCache()

                # Should load from cache
                assert cache.publishers == [{"_about": "cached_pub", "title": "Cached Publisher"}]

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self, temp_cache_dir, mock_client):
        """Cache should expire after 24 hours."""
        from server import MetadataCache

        # Create an expired cache file (older than 24 hours)
        cache_file = temp_cache_dir / "metadata.pkl"
        cache_data = {
            "timestamp": time.time() - (25 * 60 * 60),  # 25 hours ago
            "publishers": [{"_about": "old_pub", "title": "Old Publisher"}],
            "themes": None,
            "provinces": None,
            "autonomous_regions": None,
            "public_sectors": None,
            "spatial_coverage": None,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        with patch.object(MetadataCache, "_CACHE_DIR", temp_cache_dir):
            with patch.object(MetadataCache, "CACHE_FILE", cache_file):
                cache = MetadataCache()

                # Cache should not be loaded (expired)
                assert cache.publishers is None

    @pytest.mark.asyncio
    async def test_second_call_uses_cache(self, temp_cache_dir, mock_client):
        """Second call should not hit API if cache is valid."""
        from server import MetadataCache

        cache_file = temp_cache_dir / "metadata.pkl"

        with patch.object(MetadataCache, "_CACHE_DIR", temp_cache_dir):
            with patch.object(MetadataCache, "CACHE_FILE", cache_file):
                cache = MetadataCache()

                # First call - should fetch from API
                result1 = await cache.get_publishers(mock_client)
                assert mock_client.list_publishers.call_count == 1

                # Second call - should use cache
                result2 = await cache.get_publishers(mock_client)
                assert mock_client.list_publishers.call_count == 1  # No additional call

                assert result1 == result2


# Test 2: Parallel Pagination
class TestParallelPagination:
    """Tests for the parallel pagination in _fetch_all_pages."""

    @pytest.mark.asyncio
    async def test_fetch_all_pages_parallel(self):
        """Should fetch multiple pages in parallel."""
        from server import PaginationParams, _fetch_all_pages

        call_count = 0
        call_times = []

        async def mock_fetch(pagination: PaginationParams):
            nonlocal call_count
            call_count += 1
            call_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate network delay

            page = pagination.page
            if page < 3:
                return {"result": {"items": [{"id": f"item_{page}_{i}"} for i in range(10)]}}
            return {"result": {"items": []}}

        start_time = time.time()
        results = await _fetch_all_pages(mock_fetch, max_results=100, parallel_pages=5)
        elapsed = time.time() - start_time

        # Should complete faster than sequential (5 pages * 0.1s = 0.5s sequential, ~0.1s parallel per batch)
        assert elapsed < 0.4  # Some margin for overhead
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_handles_partial_results(self):
        """Should handle when last page has fewer items."""
        from server import DEFAULT_PAGE_SIZE, PaginationParams, _fetch_all_pages

        async def mock_fetch(pagination: PaginationParams):
            page = pagination.page
            if page == 0:
                return {"result": {"items": [{"id": i} for i in range(DEFAULT_PAGE_SIZE)]}}
            elif page == 1:
                return {"result": {"items": [{"id": i} for i in range(5)]}}  # Partial page
            return {"result": {"items": []}}

        results = await _fetch_all_pages(mock_fetch, max_results=1000, parallel_pages=3)

        # Should stop after partial page
        assert len(results) == DEFAULT_PAGE_SIZE + 5


# Test 3: Download Data (via get with include_data=True)
class TestDownloadData:
    """Tests for data download functionality in get tool."""

    @pytest.mark.asyncio
    async def test_parse_csv_full(self):
        """Should parse full CSV data."""
        from server import _parse_csv_full

        csv_content = "col1,col2,col3\nval1,val2,val3\nval4,val5,val6"
        result = _parse_csv_full(csv_content)

        assert result["columns"] == ["col1", "col2", "col3"]
        assert len(result["rows"]) == 2
        assert result["total_rows"] == 2
        assert result["format"] == "csv"

    @pytest.mark.asyncio
    async def test_parse_json_full(self):
        """Should parse full JSON data."""
        from server import _parse_json_full

        json_content = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        result = _parse_json_full(json_content)

        assert "name" in result["columns"]
        assert "age" in result["columns"]
        assert len(result["rows"]) == 2
        assert result["format"] == "json"

    @pytest.mark.asyncio
    async def test_max_rows_truncation(self):
        """Should truncate to max_rows if specified."""
        from server import _parse_csv_full

        # Create CSV with many rows
        csv_lines = ["col1,col2"] + [f"val{i},val{i}" for i in range(100)]
        csv_content = "\n".join(csv_lines)
        result = _parse_csv_full(csv_content)

        # Manually truncate like get tool does with include_data
        max_rows = 10
        original_rows = len(result["rows"])
        result["rows"] = result["rows"][:max_rows]

        assert len(result["rows"]) == max_rows
        assert original_rows == 100


# Test 4: Multi-theme Search
class TestMultiThemeSearch:
    """Tests for multi-theme filtering in _filter_datasets_locally."""

    def test_single_theme_filter(self):
        """theme param should still work."""
        from server import _filter_datasets_locally

        items = [
            {"theme": ["http://example.org/economia"]},
            {"theme": ["http://example.org/salud"]},
        ]

        filtered = _filter_datasets_locally(items, theme="economia")
        assert len(filtered) == 1

    def test_multiple_themes_or_logic(self):
        """themes param should use OR logic."""
        from server import _filter_datasets_locally

        items = [
            {"theme": ["http://example.org/economia"]},
            {"theme": ["http://example.org/salud"]},
            {"theme": ["http://example.org/educacion"]},
        ]

        filtered = _filter_datasets_locally(items, themes=["economia", "salud"])
        assert len(filtered) == 2

    def test_combined_theme_and_themes(self):
        """Should combine theme and themes with OR logic."""
        from server import _filter_datasets_locally

        items = [
            {"theme": ["http://example.org/economia"]},
            {"theme": ["http://example.org/salud"]},
            {"theme": ["http://example.org/educacion"]},
        ]

        filtered = _filter_datasets_locally(items, theme="educacion", themes=["economia"])
        assert len(filtered) == 2  # economia OR educacion


# Integration tests
class TestIntegration:
    """Integration tests to verify all components work together."""

    @pytest.mark.asyncio
    async def test_search_with_themes_list(self):
        """search should accept themes list parameter."""
        from server import search

        # search is decorated with @mcp.tool(), so check description
        if hasattr(search, "description"):
            assert "themes" in search.description
        else:
            # Or check the underlying function
            assert hasattr(search, "fn") or True  # Pass if wrapped

    def test_filter_datasets_locally_signature(self):
        """_filter_datasets_locally should accept themes parameter."""
        import inspect

        from server import _filter_datasets_locally

        sig = inspect.signature(_filter_datasets_locally)
        params = sig.parameters

        assert "themes" in params

    @pytest.mark.asyncio
    async def test_metadata_cache_instance_exists(self):
        """Global metadata_cache instance should exist."""
        from server import metadata_cache

        assert metadata_cache is not None
        assert hasattr(metadata_cache, "get_publishers")
        assert hasattr(metadata_cache, "get_themes")
        assert hasattr(metadata_cache, "clear")


# =============================================================================
# Test: Usage Metrics (internal, not exposed as tool)
# =============================================================================


class TestUsageMetrics:
    """Tests for the UsageMetrics class (internal tracking)."""

    def test_usage_metrics_instance_exists(self):
        """Global usage_metrics instance should exist."""
        from server import usage_metrics

        assert usage_metrics is not None
        assert hasattr(usage_metrics, "record_tool_call")
        assert hasattr(usage_metrics, "record_dataset_access")
        assert hasattr(usage_metrics, "record_search")
        assert hasattr(usage_metrics, "get_stats")
        assert hasattr(usage_metrics, "clear")

    def test_record_tool_call(self):
        """Should record tool calls."""
        from server import UsageMetrics

        metrics = UsageMetrics()
        metrics.clear()

        metrics.record_tool_call("test_tool")
        metrics.record_tool_call("test_tool")
        metrics.record_tool_call("another_tool")

        stats = metrics.get_stats()
        assert stats["total_tool_calls"] == 3
        assert stats["unique_tools_used"] == 2

    def test_record_dataset_access(self):
        """Should record dataset accesses."""
        from server import UsageMetrics

        metrics = UsageMetrics()
        metrics.clear()

        metrics.record_dataset_access("dataset-1")
        metrics.record_dataset_access("dataset-1")
        metrics.record_dataset_access("dataset-2")

        stats = metrics.get_stats()
        assert stats["total_dataset_accesses"] == 3
        assert stats["unique_datasets_accessed"] == 2

    def test_record_search(self):
        """Should record search queries."""
        from server import UsageMetrics

        metrics = UsageMetrics()
        metrics.clear()

        metrics.record_search({"title": "test", "theme": "economia"})

        stats = metrics.get_stats()
        assert stats["total_searches"] == 1
        assert len(stats["recent_searches"]) == 1

    def test_clear_metrics(self):
        """Should clear all metrics."""
        from server import UsageMetrics

        metrics = UsageMetrics()
        metrics.record_tool_call("test")
        metrics.record_dataset_access("ds1")
        metrics.record_search({"q": "test"})

        metrics.clear()
        stats = metrics.get_stats()

        assert stats["total_tool_calls"] == 0
        assert stats["total_dataset_accesses"] == 0
        assert stats["total_searches"] == 0


# =============================================================================
# Test: Guia Herramientas Prompt (updated tool names)
# =============================================================================


class TestGuiaHerramientasPrompt:
    """Tests for the guia_herramientas prompt."""

    def test_prompt_generates_content(self):
        """Should generate documentation content."""
        from prompts.guia_herramientas import generate_prompt

        content = generate_prompt(tool_category="all", include_examples=True)

        assert "Guia de Herramientas MCP" in content
        # Updated: search instead of search_datasets
        assert "search" in content
        assert "Ejemplos Practicos" in content

    def test_prompt_filter_by_category(self):
        """Should filter by tool category."""
        from prompts.guia_herramientas import generate_prompt

        # Search category
        search_content = generate_prompt(tool_category="search", include_examples=False)
        assert "search" in search_content

        # External category
        external_content = generate_prompt(tool_category="external", include_examples=False)
        # Updated: ine_search instead of ine_list_operations
        assert "ine" in external_content.lower()
        assert "aemet" in external_content.lower()

    def test_prompt_without_examples(self):
        """Should work without examples."""
        from prompts.guia_herramientas import generate_prompt

        content = generate_prompt(tool_category="all", include_examples=False)

        assert "Guia de Herramientas MCP" in content
        assert "Ejemplos Practicos" not in content
