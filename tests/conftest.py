"""Pytest fixtures and configuration for datos-gob-es-mcp tests."""

from typing import Any

import pytest
import respx

# Sample API responses for mocking
SAMPLE_DATASET_ITEM = {
    "_about": "https://datos.gob.es/apidata/catalog/dataset/test-dataset-123",
    "title": {"_value": "Test Dataset", "_lang": "es"},
    "description": {"_value": "A test dataset description", "_lang": "es"},
    "publisher": {"_about": "https://datos.gob.es/recurso/sector-publico/org/E00003901"},
    "theme": ["http://datos.gob.es/kos/sector-publico/sector/economia"],
    "keyword": [{"_value": "test", "_lang": "es"}, {"_value": "datos", "_lang": "es"}],
    "issued": "2024-01-15T10:00:00Z",
    "modified": "2024-06-20T15:30:00Z",
    "distribution": [{"_about": "dist-1", "format": "csv"}, {"_about": "dist-2", "format": "json"}],
}

SAMPLE_DISTRIBUTION_ITEM = {
    "_about": "https://datos.gob.es/apidata/catalog/distribution/dist-123",
    "title": {"_value": "CSV Distribution", "_lang": "es"},
    "accessURL": "https://example.com/data.csv",
    "format": "text/csv",
    "mediaType": "text/csv",
}

SAMPLE_THEME_ITEM = {
    "_about": "http://datos.gob.es/kos/sector-publico/sector/economia",
    "label": {"_value": "Economía", "_lang": "es"},
}

SAMPLE_PUBLISHER_ITEM = {
    "_about": "https://datos.gob.es/recurso/sector-publico/org/E00003901",
    "title": {"_value": "AEMET", "_lang": "es"},
}

SAMPLE_PROVINCE_ITEM = {
    "_about": "https://datos.gob.es/recurso/sector-publico/territorio/Provincia/Madrid",
    "label": {"_value": "Madrid", "_lang": "es"},
}

SAMPLE_REGION_ITEM = {
    "_about": "https://datos.gob.es/recurso/sector-publico/territorio/Autonomous-region/Comunidad-Madrid",
    "label": {"_value": "Comunidad de Madrid", "_lang": "es"},
}


def create_api_response(
    items: list[dict[str, Any]], page: int = 0, items_per_page: int = 10
) -> dict[str, Any]:
    """Create a mock API response with the given items."""
    return {
        "result": {
            "items": items,
            "page": page,
            "itemsPerPage": items_per_page,
            "totalResults": len(items),
        }
    }


@pytest.fixture
def mock_api():
    """Fixture to mock the datos.gob.es API using respx."""
    with respx.mock(assert_all_called=False) as mock:
        # Catch-all for dataset endpoints (using a more permissive pattern)
        mock.get(url__regex=r".*/catalog/dataset\.json.*").respond(
            json=create_api_response([SAMPLE_DATASET_ITEM])
        )
        mock.get(url__regex=r".*/catalog/dataset/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_DATASET_ITEM])
        )
        mock.get(url__regex=r".*/catalog/dataset/title/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_DATASET_ITEM])
        )
        mock.get(url__regex=r".*/catalog/dataset/publisher/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_DATASET_ITEM])
        )
        mock.get(url__regex=r".*/catalog/dataset/theme/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_DATASET_ITEM])
        )
        mock.get(url__regex=r".*/catalog/dataset/format/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_DATASET_ITEM])
        )
        mock.get(url__regex=r".*/catalog/dataset/keyword/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_DATASET_ITEM])
        )
        mock.get(url__regex=r".*/catalog/dataset/spatial/.*\.json.*").respond(
            json=create_api_response([SAMPLE_DATASET_ITEM])
        )
        mock.get(url__regex=r".*/catalog/dataset/modified/.*\.json.*").respond(
            json=create_api_response([SAMPLE_DATASET_ITEM])
        )

        # Distribution endpoints
        mock.get(url__regex=r".*/catalog/distribution\.json.*").respond(
            json=create_api_response([SAMPLE_DISTRIBUTION_ITEM])
        )
        mock.get(url__regex=r".*/catalog/distribution/dataset/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_DISTRIBUTION_ITEM])
        )
        mock.get(url__regex=r".*/catalog/distribution/format/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_DISTRIBUTION_ITEM])
        )

        # Metadata endpoints
        mock.get(url__regex=r".*/catalog/publisher\.json.*").respond(
            json=create_api_response([SAMPLE_PUBLISHER_ITEM])
        )
        mock.get(url__regex=r".*/catalog/theme\.json.*").respond(
            json=create_api_response([SAMPLE_THEME_ITEM])
        )
        mock.get(url__regex=r".*/catalog/spatial\.json.*").respond(
            json=create_api_response([{"_about": "spatial-1"}])
        )

        # NTI endpoints
        mock.get(url__regex=r".*/nti/public-sector\.json.*").respond(
            json=create_api_response([{"_about": "sector-1", "label": "Sector 1"}])
        )
        mock.get(url__regex=r".*/nti/public-sector/[^/]+\.json.*").respond(
            json=create_api_response([{"_about": "sector-1", "label": "Sector 1"}])
        )
        mock.get(url__regex=r".*/nti/territory/Province\.json.*").respond(
            json=create_api_response([SAMPLE_PROVINCE_ITEM])
        )
        mock.get(url__regex=r".*/nti/territory/Province/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_PROVINCE_ITEM])
        )
        mock.get(url__regex=r".*/nti/territory/Autonomous-region\.json.*").respond(
            json=create_api_response([SAMPLE_REGION_ITEM])
        )
        mock.get(url__regex=r".*/nti/territory/Autonomous-region/[^/]+\.json.*").respond(
            json=create_api_response([SAMPLE_REGION_ITEM])
        )
        mock.get(url__regex=r".*/nti/territory/Country/.*\.json.*").respond(
            json=create_api_response([{"_about": "spain", "label": "España"}])
        )

        yield mock


@pytest.fixture
def sample_dataset_item():
    """Return a sample dataset item for testing."""
    return SAMPLE_DATASET_ITEM.copy()


@pytest.fixture
def sample_distribution_item():
    """Return a sample distribution item for testing."""
    return SAMPLE_DISTRIBUTION_ITEM.copy()


@pytest.fixture
def sample_api_response():
    """Return a sample API response."""
    return create_api_response([SAMPLE_DATASET_ITEM])
