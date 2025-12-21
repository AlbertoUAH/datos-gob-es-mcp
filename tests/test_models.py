"""Tests for Pydantic models."""

import pytest
from server import PaginationParams, DatasetSummary, DistributionSummary


class TestPaginationParams:
    """Tests for PaginationParams model."""

    def test_default_values(self):
        """Test default pagination values."""
        params = PaginationParams()
        assert params.page == 0
        assert params.page_size == 200
        assert params.sort is None

    def test_custom_values(self):
        """Test custom pagination values."""
        params = PaginationParams(page=5, page_size=25, sort="-modified")
        assert params.page == 5
        assert params.page_size == 25
        assert params.sort == "-modified"

    def test_page_validation_negative(self):
        """Test that negative page raises validation error."""
        with pytest.raises(ValueError):
            PaginationParams(page=-1)

    def test_page_size_validation_min(self):
        """Test that page_size below 1 raises validation error."""
        with pytest.raises(ValueError):
            PaginationParams(page_size=0)

    def test_page_size_validation_max(self):
        """Test that page_size above 200 raises validation error."""
        with pytest.raises(ValueError):
            PaginationParams(page_size=201)


class TestDatasetSummary:
    """Tests for DatasetSummary model."""

    def test_from_api_item_basic(self, sample_dataset_item):
        """Test basic dataset parsing from API item."""
        summary = DatasetSummary.from_api_item(sample_dataset_item)

        assert summary.uri == sample_dataset_item["_about"]
        assert summary.title == "Test Dataset"
        assert summary.description == "A test dataset description"
        assert summary.distributions_count == 2

    def test_from_api_item_with_list_title(self):
        """Test parsing when title is a list."""
        item = {
            "_about": "test-uri",
            "title": [
                {"_value": "Title 1", "_lang": "es"},
                {"_value": "Title 2", "_lang": "en"}
            ]
        }
        summary = DatasetSummary.from_api_item(item)
        assert summary.title == ["Title 1", "Title 2"]

    def test_from_api_item_with_string_title(self):
        """Test parsing when title is a plain string."""
        item = {
            "_about": "test-uri",
            "title": "Plain Title"
        }
        summary = DatasetSummary.from_api_item(item)
        assert summary.title == "Plain Title"

    def test_from_api_item_with_string_theme(self):
        """Test parsing when theme is a string."""
        item = {
            "_about": "test-uri",
            "theme": "economia"
        }
        summary = DatasetSummary.from_api_item(item)
        assert summary.theme == ["economia"]

    def test_from_api_item_with_dict_distribution(self):
        """Test parsing when distribution is a dict (single item)."""
        item = {
            "_about": "test-uri",
            "distribution": {"_about": "dist-1", "format": "csv"}
        }
        summary = DatasetSummary.from_api_item(item)
        assert summary.distributions_count == 1

    def test_from_api_item_with_no_distribution(self):
        """Test parsing when there are no distributions."""
        item = {"_about": "test-uri"}
        summary = DatasetSummary.from_api_item(item)
        assert summary.distributions_count == 0

    def test_extract_keywords_string(self):
        """Test keyword extraction from string."""
        result = DatasetSummary._extract_keywords("single-keyword")
        assert result == ["single-keyword"]

    def test_extract_keywords_list_of_strings(self):
        """Test keyword extraction from list of strings."""
        result = DatasetSummary._extract_keywords(["keyword1", "keyword2"])
        assert result == ["keyword1", "keyword2"]

    def test_extract_keywords_list_of_dicts(self):
        """Test keyword extraction from list of dicts."""
        keywords = [
            {"_value": "keyword1", "_lang": "es"},
            {"_value": "keyword2", "_lang": "en"}
        ]
        result = DatasetSummary._extract_keywords(keywords)
        assert result == ["keyword1", "keyword2"]

    def test_extract_keywords_none(self):
        """Test keyword extraction from None."""
        result = DatasetSummary._extract_keywords(None)
        assert result is None


class TestDistributionSummary:
    """Tests for DistributionSummary model."""

    def test_from_api_item_basic(self, sample_distribution_item):
        """Test basic distribution parsing from API item."""
        summary = DistributionSummary.from_api_item(sample_distribution_item)

        assert summary.uri == sample_distribution_item["_about"]
        assert summary.title == "CSV Distribution"
        assert summary.access_url == "https://example.com/data.csv"
        assert summary.format == "text/csv"
        assert summary.media_type == "text/csv"

    def test_from_api_item_with_string_title(self):
        """Test parsing when title is a plain string."""
        item = {
            "_about": "dist-uri",
            "title": "Plain Title",
            "accessURL": "https://example.com/data.csv"
        }
        summary = DistributionSummary.from_api_item(item)
        assert summary.title == "Plain Title"

    def test_from_api_item_with_list_title(self):
        """Test parsing when title is a list."""
        item = {
            "_about": "dist-uri",
            "title": [
                {"_value": "Title 1", "_lang": "es"},
                {"_value": "Title 2", "_lang": "en"}
            ]
        }
        summary = DistributionSummary.from_api_item(item)
        assert summary.title == "Title 1"

    def test_from_api_item_minimal(self):
        """Test parsing with minimal data."""
        item = {"_about": "dist-uri"}
        summary = DistributionSummary.from_api_item(item)
        assert summary.uri == "dist-uri"
        assert summary.title is None
        assert summary.access_url is None
