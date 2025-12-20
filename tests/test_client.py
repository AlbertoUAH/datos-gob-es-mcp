"""Tests for the DatosGobClient HTTP client."""

import pytest
import httpx
import respx
from server import DatosGobClient, DatosGobClientError, PaginationParams


class TestDatosGobClient:
    """Tests for the DatosGobClient class."""

    @pytest.fixture
    def client(self):
        """Create a DatosGobClient instance."""
        return DatosGobClient()

    def test_client_initialization(self, client):
        """Test client default initialization."""
        assert client.timeout == 30.0
        assert client.BASE_URL == "https://datos.gob.es/apidata/"

    def test_client_custom_timeout(self):
        """Test client with custom timeout."""
        client = DatosGobClient(timeout=60.0)
        assert client.timeout == 60.0

    def test_build_params_none(self, client):
        """Test _build_params with no pagination."""
        params = client._build_params(None)
        assert params == {}

    def test_build_params_with_pagination(self, client):
        """Test _build_params with pagination."""
        pagination = PaginationParams(page=2, page_size=25, sort="-modified")
        params = client._build_params(pagination)

        assert params["_page"] == 2
        assert params["_pageSize"] == 25
        assert params["_sort"] == "-modified"

    def test_build_params_without_sort(self, client):
        """Test _build_params without sort."""
        pagination = PaginationParams(page=1, page_size=10)
        params = client._build_params(pagination)

        assert params["_page"] == 1
        assert params["_pageSize"] == 10
        assert "_sort" not in params

    @pytest.mark.asyncio
    async def test_list_datasets(self, client, mock_api):
        """Test list_datasets method."""
        result = await client.list_datasets()

        assert "result" in result
        assert "items" in result["result"]
        assert len(result["result"]["items"]) > 0

    @pytest.mark.asyncio
    async def test_get_dataset(self, client, mock_api):
        """Test get_dataset method."""
        result = await client.get_dataset("test-dataset-123")

        assert "result" in result
        assert "items" in result["result"]

    @pytest.mark.asyncio
    async def test_search_datasets_by_title(self, client, mock_api):
        """Test search_datasets_by_title method."""
        result = await client.search_datasets_by_title("empleo")

        assert "result" in result
        assert "items" in result["result"]

    @pytest.mark.asyncio
    async def test_get_datasets_by_publisher(self, client, mock_api):
        """Test get_datasets_by_publisher method."""
        result = await client.get_datasets_by_publisher("E00003901")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_datasets_by_theme(self, client, mock_api):
        """Test get_datasets_by_theme method."""
        result = await client.get_datasets_by_theme("economia")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_datasets_by_format(self, client, mock_api):
        """Test get_datasets_by_format method."""
        result = await client.get_datasets_by_format("csv")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_datasets_by_keyword(self, client, mock_api):
        """Test get_datasets_by_keyword method."""
        result = await client.get_datasets_by_keyword("presupuesto")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_datasets_by_spatial(self, client, mock_api):
        """Test get_datasets_by_spatial method."""
        result = await client.get_datasets_by_spatial("Autonomia", "Comunidad-Madrid")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_datasets_by_date_range(self, client, mock_api):
        """Test get_datasets_by_date_range method."""
        result = await client.get_datasets_by_date_range(
            "2024-01-01T00:00Z",
            "2024-12-31T23:59Z"
        )

        assert "result" in result

    @pytest.mark.asyncio
    async def test_list_distributions(self, client, mock_api):
        """Test list_distributions method."""
        result = await client.list_distributions()

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_distributions_by_dataset(self, client, mock_api):
        """Test get_distributions_by_dataset method."""
        result = await client.get_distributions_by_dataset("test-dataset-123")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_distributions_by_format(self, client, mock_api):
        """Test get_distributions_by_format method."""
        result = await client.get_distributions_by_format("csv")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_list_publishers(self, client, mock_api):
        """Test list_publishers method."""
        result = await client.list_publishers()

        assert "result" in result

    @pytest.mark.asyncio
    async def test_list_themes(self, client, mock_api):
        """Test list_themes method."""
        result = await client.list_themes()

        assert "result" in result

    @pytest.mark.asyncio
    async def test_list_spatial_coverage(self, client, mock_api):
        """Test list_spatial_coverage method."""
        result = await client.list_spatial_coverage()

        assert "result" in result

    @pytest.mark.asyncio
    async def test_list_public_sectors(self, client, mock_api):
        """Test list_public_sectors method."""
        result = await client.list_public_sectors()

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_public_sector(self, client, mock_api):
        """Test get_public_sector method."""
        result = await client.get_public_sector("comercio")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_list_provinces(self, client, mock_api):
        """Test list_provinces method."""
        result = await client.list_provinces()

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_province(self, client, mock_api):
        """Test get_province method."""
        result = await client.get_province("Madrid")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_list_autonomous_regions(self, client, mock_api):
        """Test list_autonomous_regions method."""
        result = await client.list_autonomous_regions()

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_autonomous_region(self, client, mock_api):
        """Test get_autonomous_region method."""
        result = await client.get_autonomous_region("Comunidad-Madrid")

        assert "result" in result

    @pytest.mark.asyncio
    async def test_get_country_spain(self, client, mock_api):
        """Test get_country_spain method."""
        result = await client.get_country_spain()

        assert "result" in result


class TestDatosGobClientErrors:
    """Tests for error handling in DatosGobClient."""

    @pytest.fixture
    def client(self):
        """Create a DatosGobClient instance."""
        return DatosGobClient(timeout=1.0)

    @pytest.mark.asyncio
    async def test_http_error(self, client):
        """Test handling of HTTP errors."""
        with respx.mock:
            respx.get(url__regex=r".*").respond(status_code=404, json={"error": "Not found"})

            with pytest.raises(DatosGobClientError) as exc_info:
                await client.list_datasets()

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_timeout_error(self, client):
        """Test handling of timeout errors."""
        with respx.mock:
            respx.get(url__regex=r".*").mock(side_effect=httpx.TimeoutException("Timeout"))

            with pytest.raises(DatosGobClientError) as exc_info:
                await client.list_datasets()

            assert "timed out" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_request_error(self, client):
        """Test handling of request errors."""
        with respx.mock:
            respx.get(url__regex=r".*").mock(side_effect=httpx.RequestError("Connection failed"))

            with pytest.raises(DatosGobClientError) as exc_info:
                await client.list_datasets()

            assert "failed" in exc_info.value.message.lower()


class TestDatosGobClientError:
    """Tests for the DatosGobClientError exception."""

    def test_error_with_message_only(self):
        """Test error with message only."""
        error = DatosGobClientError("Test error")
        assert error.message == "Test error"
        assert error.status_code is None
        assert str(error) == "Test error"

    def test_error_with_status_code(self):
        """Test error with status code."""
        error = DatosGobClientError("Not found", status_code=404)
        assert error.message == "Not found"
        assert error.status_code == 404
