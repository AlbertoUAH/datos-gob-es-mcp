"""AEMET (Agencia Estatal de Meteorología) API integration.

API Documentation: https://opendata.aemet.es/dist/index.html
Base URL: https://opendata.aemet.es/opendata/api/

Requires free API key from: https://opendata.aemet.es/centrodedescargas/altaUsuario
"""

import json
import os
from typing import Any

from dotenv import load_dotenv

from core import AEMETClientError, HTTPClient, get_logger, handle_api_error
from core.config import (
    AEMET_BASE_URL,
    AEMET_MAX_FORECAST_DAYS,
    AEMET_MAX_MUNICIPALITIES,
    AEMET_MAX_OBSERVATIONS,
    HTTP_DEFAULT_TIMEOUT,
)

# Load environment variables
load_dotenv()

logger = get_logger("aemet")


class AEMETClient:
    """Async HTTP client for the AEMET OpenData API.

    Uses HTTPClient for automatic logging and rate limiting.
    Note: AEMET has a special two-step request process (get URL, then fetch data),
    so it doesn't inherit from BaseAPIClient directly.
    """

    BASE_URL = AEMET_BASE_URL
    API_NAME = "aemet"
    ERROR_CLASS = AEMETClientError

    def __init__(self, api_key: str | None = None, timeout: float = HTTP_DEFAULT_TIMEOUT):
        self.api_key = api_key or os.getenv("AEMET_API_KEY")
        self.timeout = timeout
        self.http = HTTPClient(self.API_NAME, self.BASE_URL, timeout)

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key."""
        if not self.api_key:
            raise AEMETClientError(
                "AEMET API key not configured. Set AEMET_API_KEY environment variable "
                "or get a free key at https://opendata.aemet.es/centrodedescargas/altaUsuario"
            )
        return {"api_key": self.api_key}

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make an async HTTP request to the AEMET API with logging and rate limiting.

        AEMET API returns a URL to fetch the actual data, so this method
        handles the two-step process.
        """
        headers = self._get_headers()

        try:
            # First request gets the data URL (with rate limiting)
            response = await self.http.get(endpoint, params=params, headers=headers)
            result = response.json()

            # Check for API errors
            if result.get("estado") != 200:
                raise AEMETClientError(
                    result.get("descripcion", "Unknown error"),
                    status_code=result.get("estado"),
                )

            # Second request fetches actual data (no rate limiting needed, different domain)
            data_url = result.get("datos")
            if data_url:
                logger.debug("fetching_data_url", url=data_url)
                # Use a separate HTTPClient for the data URL (no rate limiting)
                data_http = HTTPClient("aemet_data", "", self.timeout, rate_limit=False)
                data_response = await data_http.get(data_url, headers=headers)
                # AEMET returns data in Latin-1 encoding, not UTF-8
                content = data_response.content.decode("latin-1")
                return json.loads(content)

            return result

        except AEMETClientError:
            raise
        except Exception as e:
            if hasattr(e, "status_code"):
                raise AEMETClientError(str(e), status_code=e.status_code) from e
            raise AEMETClientError(str(e)) from e

    async def get_municipalities(self) -> list[dict[str, Any]]:
        """Get list of all municipalities with weather data."""
        return await self._request("maestro/municipios")

    async def get_forecast_daily(self, municipality_code: str) -> list[dict[str, Any]]:
        """Get daily weather forecast for a municipality.

        Args:
            municipality_code: 5-digit municipality code (e.g., '28079' for Madrid)
        """
        return await self._request(f"prediccion/especifica/municipio/diaria/{municipality_code}")

    async def get_forecast_hourly(self, municipality_code: str) -> list[dict[str, Any]]:
        """Get hourly weather forecast for a municipality.

        Args:
            municipality_code: 5-digit municipality code
        """
        return await self._request(f"prediccion/especifica/municipio/horaria/{municipality_code}")

    async def get_stations(self) -> list[dict[str, Any]]:
        """Get list of all weather observation stations."""
        return await self._request("valores/climatologicos/inventarioestaciones/todasestaciones")

    async def get_observations(self, station_id: str = None) -> list[dict[str, Any]]:
        """Get current weather observations.

        Args:
            station_id: Optional station ID to filter (e.g., '3129' for Madrid-Retiro)
        """
        if station_id:
            return await self._request(f"observacion/convencional/datos/estacion/{station_id}")
        return await self._request("observacion/convencional/todas")

    async def get_beach_forecast(self, beach_id: str) -> dict[str, Any]:
        """Get beach weather forecast.

        Args:
            beach_id: Beach identifier
        """
        return await self._request(f"prediccion/especifica/playa/{beach_id}")

    async def get_fire_risk(self) -> list[dict[str, Any]]:
        """Get forest fire risk predictions."""
        return await self._request("incendios/mapasriesgo/estimado/area")

    async def get_uv_index(self, day: int = 0) -> list[dict[str, Any]]:
        """Get UV index predictions.

        Args:
            day: Day offset (0=today, 1=tomorrow, etc., max 4)
        """
        return await self._request(f"prediccion/especifica/uvi/{day}")


# Global client instance (will be configured with API key from env)
aemet_client = AEMETClient()


def _handle_error(e: Exception) -> str:
    """Format error message."""
    return handle_api_error(e, context="aemet_operation", logger_name="aemet")


def _format_forecast(forecast_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Format forecast data for cleaner output."""
    if not forecast_data:
        return {"error": "No forecast data available"}

    forecast = forecast_data[0]
    prediccion = forecast.get("prediccion", {})
    dias = prediccion.get("dia", [])

    formatted_days = []
    for dia in dias[:AEMET_MAX_FORECAST_DAYS]:  # Max forecast days
        formatted_days.append(
            {
                "fecha": dia.get("fecha"),
                "temp_max": dia.get("temperatura", {}).get("maxima")
                if isinstance(dia.get("temperatura"), dict)
                else None,
                "temp_min": dia.get("temperatura", {}).get("minima")
                if isinstance(dia.get("temperatura"), dict)
                else None,
                "estado_cielo": dia.get("estadoCielo", [{}])[0].get("descripcion")
                if dia.get("estadoCielo")
                else None,
                "prob_precipitacion": dia.get("probPrecipitacion", [{}])[0].get("value")
                if dia.get("probPrecipitacion")
                else None,
                "viento": dia.get("viento", [{}])[0] if dia.get("viento") else None,
            }
        )

    return {
        "municipio": forecast.get("nombre"),
        "provincia": forecast.get("provincia"),
        "elaborado": forecast.get("elaborado"),
        "dias": formatted_days,
    }


def register_aemet_tools(mcp):
    """Register AEMET tools with the MCP server."""

    @mcp.tool()
    async def aemet_get_forecast(municipality_code: str) -> str:
        """Get weather forecast for a Spanish municipality.

        Retrieve the daily weather forecast for any Spanish municipality.
        Includes temperature, precipitation probability, sky conditions, and wind.

        Requires AEMET_API_KEY environment variable (free from AEMET OpenData).

        Args:
            municipality_code: 5-digit municipality code (e.g., '28079' for Madrid,
                '08019' for Barcelona, '41091' for Sevilla). Use aemet_list_locations
                to find codes.

        Returns:
            JSON with 7-day forecast including temperatures, precipitation, and conditions.
        """
        try:
            data = await aemet_client.get_forecast_daily(municipality_code)
            formatted = _format_forecast(data)
            return json.dumps(formatted, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def aemet_get_observations(station_id: str | None = None) -> str:
        """Get current weather observations from AEMET stations.

        Retrieve current weather data from meteorological stations.
        Returns temperature, humidity, pressure, wind, and precipitation.

        Args:
            station_id: Optional station ID (e.g., '3129' for Madrid-Retiro).
                If not provided, returns data from all stations.

        Returns:
            JSON with current weather observations.
        """
        try:
            data = await aemet_client.get_observations(station_id)

            # Format observations
            observations = []
            for obs in (
                data[:AEMET_MAX_OBSERVATIONS] if isinstance(data, list) else [data]
            ):  # Limit results
                observations.append(
                    {
                        "estacion": obs.get("ubi"),
                        "id": obs.get("idema"),
                        "fecha": obs.get("fint"),
                        "temperatura": obs.get("ta"),
                        "temp_max": obs.get("tamax"),
                        "temp_min": obs.get("tamin"),
                        "humedad": obs.get("hr"),
                        "presion": obs.get("pres"),
                        "viento_vel": obs.get("vv"),
                        "viento_dir": obs.get("dv"),
                        "precipitacion": obs.get("prec"),
                    }
                )

            output = {
                "total_stations": len(observations),
                "observations": observations,
            }
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def aemet_list_locations(location_type: str = "all") -> str:
        """List AEMET locations: municipalities for forecasts or stations for observations.

        Args:
            location_type: Type of locations to list:
                - 'municipalities': List municipalities (use codes with aemet_get_forecast)
                - 'stations': List observation stations (use IDs with aemet_get_observations)
                - 'all': List both (default)

        Returns:
            JSON with locations. Use municipality codes for forecasts, station IDs for observations.
        """
        try:
            output = {}

            if location_type in ("municipalities", "all"):
                municipalities = await aemet_client.get_municipalities()
                output["municipalities"] = {
                    "total": len(municipalities),
                    "items": [
                        {
                            "codigo": m.get("id"),
                            "nombre": m.get("nombre"),
                            "provincia": m.get("id", "")[:2] if m.get("id") else None,
                        }
                        for m in municipalities[:AEMET_MAX_MUNICIPALITIES]
                    ],
                }

            if location_type in ("stations", "all"):
                stations = await aemet_client.get_stations()
                output["stations"] = {
                    "total": len(stations),
                    "items": [
                        {
                            "id": s.get("indicativo"),
                            "nombre": s.get("nombre"),
                            "provincia": s.get("provincia"),
                        }
                        for s in stations
                    ],
                }

            if not output:
                return json.dumps(
                    {
                        "error": f"Invalid location_type: {location_type}. Use 'municipalities', 'stations', or 'all'."
                    },
                    ensure_ascii=False,
                )

            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)
