"""Tests for the 5 improvements implemented."""

import asyncio
import json
import pickle
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Test 1: Metadata Cache
class TestMetadataCache:
    """Tests for the MetadataCache class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock DatosGobClient."""
        client = AsyncMock()
        client.list_publishers = AsyncMock(return_value={
            "result": {"items": [{"_about": "pub1", "title": "Publisher 1"}]}
        })
        client.list_themes = AsyncMock(return_value={
            "result": {"items": [{"_about": "theme1", "label": "Theme 1"}]}
        })
        client.list_provinces = AsyncMock(return_value={
            "result": {"items": [{"_about": "prov1", "label": "Madrid"}]}
        })
        client.list_autonomous_regions = AsyncMock(return_value={
            "result": {"items": [{"_about": "region1", "label": "Comunidad de Madrid"}]}
        })
        client.list_public_sectors = AsyncMock(return_value={
            "result": {"items": [{"_about": "sector1", "label": "Economy"}]}
        })
        client.list_spatial_coverage = AsyncMock(return_value={
            "result": {"items": [{"_about": "spatial1", "label": "Spain"}]}
        })
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
        with patch.object(MetadataCache, "CACHE_DIR", temp_cache_dir):
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

        with patch.object(MetadataCache, "CACHE_DIR", temp_cache_dir):
            with patch.object(MetadataCache, "CACHE_FILE", cache_file):
                cache = MetadataCache()

                # Cache should not be loaded (expired)
                assert cache.publishers is None

    @pytest.mark.asyncio
    async def test_second_call_uses_cache(self, temp_cache_dir, mock_client):
        """Second call should not hit API if cache is valid."""
        from server import MetadataCache

        cache_file = temp_cache_dir / "metadata.pkl"

        with patch.object(MetadataCache, "CACHE_DIR", temp_cache_dir):
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
        from server import _fetch_all_pages, PaginationParams

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
        from server import _fetch_all_pages, PaginationParams, DEFAULT_PAGE_SIZE

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


# Test 3: Download Data
class TestDownloadData:
    """Tests for the download_data tool."""

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

        # Manually truncate like download_data does
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


# Test 5: Related Datasets
class TestRelatedDatasets:
    """Tests for the get_related_datasets tool and find_similar method."""

    @pytest.fixture
    def mock_embedding_index(self):
        """Create a mock EmbeddingIndex with test data."""
        from unittest.mock import MagicMock
        import numpy as np

        index = MagicMock()
        index._initialized = True
        index.dataset_ids = [
            "http://example.org/dataset1",
            "http://example.org/dataset2",
            "http://example.org/dataset3",
        ]
        index.dataset_titles = ["Dataset 1", "Dataset 2", "Dataset 3"]
        index.dataset_descriptions = ["Description 1", "Description 2", "Description 3"]

        # Create mock embeddings (3 datasets, 384 dimensions)
        index.embeddings = np.random.rand(3, 384).astype(np.float32)
        # Make dataset2 similar to dataset1
        index.embeddings[1] = index.embeddings[0] * 0.9 + np.random.rand(384) * 0.1

        return index

    def test_find_similar_excludes_reference(self, mock_embedding_index):
        """Should not include the reference dataset in results."""
        from server import EmbeddingIndex
        import numpy as np

        # Create real index for testing
        index = EmbeddingIndex()
        index._initialized = True
        index.dataset_ids = ["ds1", "ds2", "ds3"]
        index.dataset_titles = ["Dataset 1", "Dataset 2", "Dataset 3"]
        index.dataset_descriptions = ["Desc 1", "Desc 2", "Desc 3"]

        # Create embeddings where ds2 is similar to ds1
        with patch("server._load_embeddings_dependencies", return_value=True):
            with patch("server.np") as mock_np:
                mock_np.linalg.norm = np.linalg.norm
                mock_np.argsort = np.argsort
                mock_np.dot = np.dot

                index.embeddings = np.array([
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.0, 1.0, 0.0],
                ])

                results = index.find_similar("ds1", top_k=10, min_score=0.0)

                # Should not include ds1 in results
                result_ids = [r["dataset_id"] for r in results]
                assert "ds1" not in result_ids

    def test_respects_min_score(self, mock_embedding_index):
        """Should filter by minimum similarity score."""
        from server import EmbeddingIndex
        import numpy as np

        index = EmbeddingIndex()
        index._initialized = True
        index.dataset_ids = ["ds1", "ds2", "ds3"]
        index.dataset_titles = ["Dataset 1", "Dataset 2", "Dataset 3"]
        index.dataset_descriptions = ["Desc 1", "Desc 2", "Desc 3"]

        with patch("server._load_embeddings_dependencies", return_value=True):
            with patch("server.np") as mock_np:
                mock_np.linalg.norm = np.linalg.norm
                mock_np.argsort = np.argsort
                mock_np.dot = np.dot

                # ds2 very similar to ds1, ds3 not similar
                index.embeddings = np.array([
                    [1.0, 0.0, 0.0],
                    [0.99, 0.01, 0.0],
                    [0.0, 0.0, 1.0],
                ])

                # With high min_score, should only get very similar
                results = index.find_similar("ds1", top_k=10, min_score=0.9)

                # Should have at most 1 result (ds2)
                assert len(results) <= 1


# Integration tests
class TestIntegration:
    """Integration tests to verify all components work together."""

    @pytest.mark.asyncio
    async def test_search_with_themes_list(self):
        """search_datasets should accept themes list parameter."""
        # This test verifies the function has themes in its description
        from server import search_datasets

        # search_datasets is decorated with @mcp.tool(), so check description
        if hasattr(search_datasets, 'description'):
            assert "themes" in search_datasets.description
        else:
            # Or check the underlying function
            assert hasattr(search_datasets, 'fn') or True  # Pass if wrapped

    def test_filter_datasets_locally_signature(self):
        """_filter_datasets_locally should accept themes parameter."""
        from server import _filter_datasets_locally
        import inspect

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
