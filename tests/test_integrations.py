"""Tests for external API integrations (INE, AEMET, BOE)."""

import pytest
import respx

from integrations.aemet import AEMETClient, AEMETClientError
from integrations.boe import BOEClient, BOEClientError
from integrations.ine import INEClient, INEClientError

# =============================================================================
# INE Tests
# =============================================================================


class TestINEClient:
    """Tests for INE API client."""

    @pytest.fixture
    def ine_client(self):
        return INEClient(timeout=10.0)

    @pytest.fixture
    def mock_ine_api(self):
        """Mock INE API responses."""
        with respx.mock(assert_all_called=False) as mock:
            base_url = "https://servicios.ine.es/wstempus/js/"

            # Mock operations list
            mock.get(f"{base_url}ES/OPERACIONES_DISPONIBLES").respond(
                json=[
                    {"Id": "1", "Nombre": "Encuesta de Poblacion Activa", "Cod_IOE": "EPA"},
                    {"Id": "2", "Nombre": "Indice de Precios de Consumo", "Cod_IOE": "IPC"},
                ]
            )

            # Mock operation details
            mock.get(url__regex=r".*/ES/OPERACION/\d+").respond(
                json={"Id": "1", "Nombre": "Test Operation"}
            )

            # Mock tables list
            mock.get(url__regex=r".*/ES/TABLAS_OPERACION/\d+").respond(
                json=[
                    {"Id": "100", "Nombre": "Tabla 1", "Codigo": "T1"},
                    {"Id": "101", "Nombre": "Tabla 2", "Codigo": "T2"},
                ]
            )

            # Mock table data
            mock.get(url__regex=r".*/ES/DATOS_TABLA/\d+").respond(
                json=[
                    {"Nombre": "Dato 1", "Valor": 100, "Fecha": "2024-01-01"},
                    {"Nombre": "Dato 2", "Valor": 200, "Fecha": "2024-02-01"},
                ]
            )

            yield mock

    @pytest.mark.asyncio
    async def test_list_operations(self, ine_client, mock_ine_api):
        """Test listing INE operations."""
        operations = await ine_client.list_operations()

        assert len(operations) == 2
        assert operations[0]["Nombre"] == "Encuesta de Poblacion Activa"

    @pytest.mark.asyncio
    async def test_search_operations(self, ine_client, mock_ine_api):
        """Test searching INE operations."""
        results = await ine_client.search_operations("Poblacion")

        assert len(results) == 1
        assert "Poblacion" in results[0]["Nombre"]

    @pytest.mark.asyncio
    async def test_list_tables(self, ine_client, mock_ine_api):
        """Test listing tables for an operation."""
        tables = await ine_client.list_tables("1")

        assert len(tables) == 2
        assert tables[0]["Id"] == "100"

    @pytest.mark.asyncio
    async def test_get_table_data(self, ine_client, mock_ine_api):
        """Test getting table data."""
        data = await ine_client.get_table_data("100", n_last=5)

        assert len(data) == 2
        assert data[0]["Valor"] == 100

    @pytest.mark.asyncio
    async def test_ine_error_handling(self, ine_client):
        """Test INE client error handling."""
        with respx.mock:
            respx.get(url__regex=r".*").respond(status_code=500, json={"error": "Server error"})

            with pytest.raises(INEClientError) as exc_info:
                await ine_client.list_operations()

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_ine_empty_response_handling(self, ine_client):
        """Test INE client handles empty responses gracefully.

        The INE API returns empty body (not []) for operations without tables.
        This should return an empty list, not raise a JSON decode error.
        """
        with respx.mock:
            base_url = "https://servicios.ine.es/wstempus/js/"
            # Empty response body (INE quirk for operations without tables)
            respx.get(f"{base_url}ES/TABLAS_OPERACION/99999").respond(
                status_code=200,
                content=b"",
            )

            tables = await ine_client.list_tables("99999")

            assert tables == []
            assert isinstance(tables, list)


# =============================================================================
# AEMET Tests
# =============================================================================


class TestAEMETClient:
    """Tests for AEMET API client."""

    @pytest.fixture
    def aemet_client(self):
        return AEMETClient(api_key="test-key", timeout=10.0)

    @pytest.fixture
    def mock_aemet_api(self):
        """Mock AEMET API responses."""
        with respx.mock(assert_all_called=False) as mock:
            base_url = "https://opendata.aemet.es/opendata/api/"

            # AEMET returns a URL to fetch data from
            mock.get(url__regex=r".*/prediccion/especifica/municipio/diaria/\d+").respond(
                json={
                    "estado": 200,
                    "datos": "https://opendata.aemet.es/data/forecast123",
                    "metadatos": "https://opendata.aemet.es/meta/123",
                }
            )

            # Mock the data URL
            mock.get("https://opendata.aemet.es/data/forecast123").respond(
                json=[
                    {
                        "nombre": "Madrid",
                        "provincia": "Madrid",
                        "elaborado": "2024-12-26T10:00:00",
                        "prediccion": {"dia": []},
                    }
                ]
            )

            # Mock municipalities
            mock.get(f"{base_url}maestro/municipios").respond(
                json={
                    "estado": 200,
                    "datos": "https://opendata.aemet.es/data/municipios",
                }
            )
            mock.get("https://opendata.aemet.es/data/municipios").respond(
                json=[
                    {"id": "28079", "nombre": "Madrid"},
                    {"id": "08019", "nombre": "Barcelona"},
                ]
            )

            # Mock stations
            mock.get(url__regex=r".*/valores/climatologicos/inventarioestaciones/.*").respond(
                json={
                    "estado": 200,
                    "datos": "https://opendata.aemet.es/data/stations",
                }
            )
            mock.get("https://opendata.aemet.es/data/stations").respond(
                json=[
                    {"indicativo": "3129", "nombre": "Madrid-Retiro", "provincia": "Madrid"},
                ]
            )

            yield mock

    @pytest.mark.asyncio
    async def test_get_forecast(self, aemet_client, mock_aemet_api):
        """Test getting weather forecast."""
        forecast = await aemet_client.get_forecast_daily("28079")

        assert len(forecast) == 1
        assert forecast[0]["nombre"] == "Madrid"

    @pytest.mark.asyncio
    async def test_get_municipalities(self, aemet_client, mock_aemet_api):
        """Test getting municipalities list."""
        municipalities = await aemet_client.get_municipalities()

        assert len(municipalities) == 2
        assert municipalities[0]["id"] == "28079"

    @pytest.mark.asyncio
    async def test_get_stations(self, aemet_client, mock_aemet_api):
        """Test getting weather stations."""
        stations = await aemet_client.get_stations()

        assert len(stations) == 1
        assert stations[0]["indicativo"] == "3129"

    @pytest.mark.asyncio
    async def test_aemet_no_api_key(self, monkeypatch):
        """Test AEMET client without API key."""
        # Remove AEMET_API_KEY from environment for this test
        monkeypatch.delenv("AEMET_API_KEY", raising=False)
        client = AEMETClient(api_key=None)

        with pytest.raises(AEMETClientError) as exc_info:
            client._get_headers()

        assert "API key not configured" in str(exc_info.value)


# =============================================================================
# BOE Tests
# =============================================================================


class TestBOEClient:
    """Tests for BOE API client."""

    @pytest.fixture
    def boe_client(self):
        return BOEClient(timeout=10.0)

    @pytest.fixture
    def mock_boe_api(self):
        """Mock BOE API responses."""
        with respx.mock(assert_all_called=False) as mock:
            # Mock summary
            mock.get(url__regex=r".*/boe/sumario/\d+").respond(
                json={
                    "data": {
                        "sumario": {
                            "metadatos": {
                                "fecha_publicacion": "2024-12-26",
                                "pub_numero": "312",
                                "numero_paginas": 100,
                            },
                            "diario": [
                                {
                                    "sumario_nombre": "Seccion I",
                                    "seccion": [{"departamento": "Ministerio de Hacienda"}],
                                }
                            ],
                        }
                    }
                }
            )

            # Mock document
            mock.get(url__regex=r".*/boe/documento/.*").respond(
                json={
                    "data": {
                        "documento": {
                            "metadatos": {
                                "identificador": "BOE-A-2024-12345",
                                "titulo": "Real Decreto de ejemplo",
                                "fecha_publicacion": "2024-12-26",
                            },
                            "analisis": {"materias": [], "notas": []},
                        }
                    }
                }
            )

            yield mock

    @pytest.mark.asyncio
    async def test_get_summary(self, boe_client, mock_boe_api):
        """Test getting BOE summary."""
        summary = await boe_client.get_summary("20241226")

        assert "data" in summary
        assert summary["data"]["sumario"]["metadatos"]["pub_numero"] == "312"

    @pytest.mark.asyncio
    async def test_get_document(self, boe_client, mock_boe_api):
        """Test getting BOE document."""
        doc = await boe_client.get_document("BOE-A-2024-12345")

        assert "data" in doc
        assert doc["data"]["documento"]["metadatos"]["identificador"] == "BOE-A-2024-12345"

    @pytest.mark.asyncio
    async def test_boe_error_handling(self, boe_client):
        """Test BOE client error handling."""
        with respx.mock:
            respx.get(url__regex=r".*").respond(status_code=404, json={"error": "Not found"})

            with pytest.raises(BOEClientError) as exc_info:
                await boe_client.get_summary("20241226")

            assert exc_info.value.status_code == 404
