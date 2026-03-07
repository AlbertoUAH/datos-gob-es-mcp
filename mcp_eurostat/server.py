"""MCP server for Eurostat open data (European Statistics Office).

API: https://ec.europa.eu/eurostat/api/dissemination/
No authentication required.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP

from core import BaseAPIClient, get_logger, handle_api_error, setup_logging
from core.exceptions import APIClientError

setup_logging()
logger = get_logger("eurostat")


class EurostatClientError(APIClientError):
    """Error raised by EurostatClient."""


class EurostatClient(BaseAPIClient):
    """Client for the Eurostat SDMX dissemination API."""

    BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/"
    API_NAME = "eurostat"
    ERROR_CLASS = EurostatClientError

    async def search_datasets(
        self, query: str, theme: str | None = None, max_results: int = 20
    ) -> dict[str, Any]:
        """Search Eurostat dataset catalogue."""
        params: dict[str, Any] = {
            "query": query,
            "type": "dataset",
            "lang": "en",
            "hits": max_results,
        }
        if theme:
            params["theme"] = theme
        return await self._request("statistics/1.0/search", params=params)

    async def get_dataset_metadata(self, dataset_code: str) -> dict[str, Any]:
        """Get dataset metadata / data structure from SDMX."""
        return await self._request(
            f"statistics/1.0/sdmx/2.1/datastructure/ESTAT/{dataset_code.upper()}"
        )

    async def get_dataset_data(
        self,
        dataset_code: str,
        geo: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
        max_rows: int = 100,
    ) -> dict[str, Any]:
        """Download dataset data in JSON/SDMX-JSON format."""
        params: dict[str, Any] = {"lang": "en", "format": "JSON"}
        if geo:
            params["geo"] = geo
        if time_from:
            params["sinceTimePeriod"] = time_from
        if time_to:
            params["untilTimePeriod"] = time_to

        data = await self._request(
            f"statistics/1.0/data/{dataset_code.upper()}", params=params
        )
        return data


_client = EurostatClient()


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "eurostat",
    instructions=(
        "Acceso a datos estadísticos de Eurostat (Oficina de Estadística de la Unión Europea). "
        "Proporciona datos comparativos de todos los países de la UE y candidatos. "
        "Usa eurostat_search para buscar datasets, eurostat_get para metadatos y "
        "eurostat_download para obtener los datos."
    ),
)


def _handle_error(e: Exception) -> str:
    return handle_api_error(e, context="eurostat_operation", logger_name="eurostat")


def _format_search_results(data: dict[str, Any], query: str) -> str:
    """Format Eurostat search results."""
    results = data.get("results", data.get("hits", []))

    if isinstance(results, dict):
        # Some API responses wrap results differently
        results = results.get("item", results.get("items", []))

    datasets = []
    for item in results[:50]:  # Safety limit
        dataset = {
            "code": item.get("code") or item.get("id"),
            "title": item.get("title") or item.get("label"),
            "description": (item.get("shortDescription") or item.get("metadata", {}).get("shortDescription", ""))[:300],
            "last_update": item.get("lastUpdate") or item.get("lastModifiedDate"),
            "unit": item.get("unit"),
        }
        # Remove None values
        dataset = {k: v for k, v in dataset.items() if v is not None}
        datasets.append(dataset)

    return json.dumps(
        {"query": query, "total_results": len(datasets), "datasets": datasets},
        ensure_ascii=False,
        indent=2,
    )


def _format_metadata(data: dict[str, Any], dataset_code: str) -> str:
    """Format Eurostat dataset metadata."""
    # SDMX structure response
    structure = data.get("Structure", data)

    output: dict[str, Any] = {"dataset_code": dataset_code.upper()}

    # Try to extract dimensions from SDMX structure
    try:
        structures = structure.get("Structures", {})
        dsds = structures.get("DataStructures", {})
        dsd_list = dsds.get("DataStructure", [])
        if isinstance(dsd_list, dict):
            dsd_list = [dsd_list]

        for dsd in dsd_list:
            dims = []
            components = dsd.get("DataStructureComponents", {})
            dim_list = components.get("DimensionList", {}).get("Dimension", [])
            if isinstance(dim_list, dict):
                dim_list = [dim_list]
            for dim in dim_list:
                dim_id = dim.get("@id") or dim.get("id")
                if dim_id:
                    dims.append(dim_id)
            if dims:
                output["dimensions"] = dims
            break
    except Exception:
        pass

    # If structure parsing failed, return raw structure summary
    if "dimensions" not in output:
        output["raw_structure"] = str(structure)[:500]

    return json.dumps(output, ensure_ascii=False, indent=2)


def _format_data(data: dict[str, Any], dataset_code: str, max_rows: int) -> str:
    """Format Eurostat data response (SDMX-JSON)."""
    output: dict[str, Any] = {"dataset_code": dataset_code.upper()}

    try:
        # SDMX-JSON format
        dimension_info = data.get("dimension", {})
        values = data.get("value", {})

        # Extract dimension labels
        dimensions: dict[str, list[str]] = {}
        for dim_name, dim_data in dimension_info.items():
            if isinstance(dim_data, dict):
                category = dim_data.get("category", {})
                label = category.get("label", {})
                if label:
                    dimensions[dim_name] = list(label.values())[:20]

        output["dimensions"] = dimensions
        output["value_count"] = len(values)

        # Sample values
        sample_values = dict(list(values.items())[:max_rows])
        output["sample_values"] = sample_values

    except Exception:
        output["raw_data"] = str(data)[:1000]

    return json.dumps(output, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def eurostat_search(
    query: str,
    theme: str | None = None,
    max_results: int = 20,
) -> str:
    """Search datasets in the Eurostat catalogue.

    Args:
        query: Search query (e.g., 'unemployment youth', 'GDP per capita').
        theme: Optional theme filter (e.g., 'economy', 'population').
        max_results: Maximum results to return (default 20, max 100).

    Returns:
        JSON with matching Eurostat datasets (code, title, description).
    """
    try:
        max_results = min(max_results, 100)
        data = await _client.search_datasets(query, theme=theme, max_results=max_results)
        return _format_search_results(data, query)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def eurostat_get(dataset_code: str) -> str:
    """Get metadata for a Eurostat dataset (dimensions, units, periods).

    Args:
        dataset_code: Eurostat dataset code (e.g., 'UNE_RT_A', 'NAMA_10_GDP').

    Returns:
        JSON with dataset dimensions and structure.
    """
    try:
        data = await _client.get_dataset_metadata(dataset_code)
        return _format_metadata(data, dataset_code)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def eurostat_download(
    dataset_code: str,
    geo: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    max_rows: int = 100,
) -> str:
    """Download data from a Eurostat dataset.

    Args:
        dataset_code: Eurostat dataset code (e.g., 'UNE_RT_A', 'NAMA_10_GDP').
        geo: Country/region code filter (e.g., 'ES' for Spain, 'EU27_2020' for EU).
        time_from: Start period filter (e.g., '2010', '2020-Q1').
        time_to: End period filter (e.g., '2023', '2023-Q4').
        max_rows: Maximum data rows to return (default 100).

    Returns:
        JSON with dataset values indexed by dimension combination.
    """
    try:
        max_rows = min(max_rows, 1000)
        data = await _client.get_dataset_data(
            dataset_code,
            geo=geo,
            time_from=time_from,
            time_to=time_to,
            max_rows=max_rows,
        )
        return _format_data(data, dataset_code, max_rows)
    except Exception as e:
        return _handle_error(e)
