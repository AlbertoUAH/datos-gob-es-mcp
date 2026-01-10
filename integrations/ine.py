"""INE (Instituto Nacional de Estadística) API integration.

API Documentation: https://www.ine.es/dyngs/DataLab/es/manual.html
Base URL: https://servicios.ine.es/wstempus/js/
"""

import json
from typing import Any

from core import get_logger, BaseAPIClient, INEClientError, handle_api_error
from core.config import (
    INE_BASE_URL,
    DEFAULT_PAGE_SIZE,
    INE_MAX_TABLES,
    INE_MAX_DATA_RECORDS,
    INE_DEFAULT_NLAST,
)

logger = get_logger("ine")


class INEClient(BaseAPIClient):
    """Async HTTP client for the INE API.

    Uses BaseAPIClient for automatic logging and rate limiting.
    """

    BASE_URL = INE_BASE_URL
    API_NAME = "ine"
    ERROR_CLASS = INEClientError

    async def list_operations(self) -> list[dict[str, Any]]:
        """List all statistical operations available in INE."""
        return await self._request("ES/OPERACIONES_DISPONIBLES")

    async def search_operations(self, query: str) -> list[dict[str, Any]]:
        """Search statistical operations by name."""
        operations = await self.list_operations()
        query_lower = query.lower()
        return [
            op for op in operations
            if query_lower in op.get("Nombre", "").lower()
        ]

    async def get_operation(self, operation_id: str) -> dict[str, Any]:
        """Get details of a specific operation."""
        return await self._request(f"ES/OPERACION/{operation_id}")

    async def list_tables(self, operation_id: str) -> list[dict[str, Any]]:
        """List all tables for a statistical operation."""
        return await self._request(f"ES/TABLAS_OPERACION/{operation_id}")

    async def get_table_metadata(self, table_id: str) -> dict[str, Any]:
        """Get metadata for a specific table."""
        return await self._request(f"ES/METADATATABLA/{table_id}")

    async def get_table_data(
        self,
        table_id: str,
        n_last: int | None = None,
        date: str | None = None
    ) -> list[dict[str, Any]]:
        """Get data from a specific table.

        Args:
            table_id: Table identifier
            n_last: Get last N periods (optional)
            date: Specific date in format YYYYMMDD (optional)
        """
        params = {}
        if n_last:
            params["nult"] = n_last
        if date:
            params["date"] = date

        return await self._request(f"ES/DATOS_TABLA/{table_id}", params if params else None)

    async def list_series(self, operation_id: str) -> list[dict[str, Any]]:
        """List all time series for an operation."""
        return await self._request(f"ES/SERIES_OPERACION/{operation_id}")

    async def get_series_data(
        self,
        series_id: str,
        n_last: int | None = None,
        date_start: str | None = None,
        date_end: str | None = None
    ) -> list[dict[str, Any]]:
        """Get data from a specific time series.

        Args:
            series_id: Series identifier (COD)
            n_last: Get last N periods
            date_start: Start date YYYYMMDD
            date_end: End date YYYYMMDD
        """
        params = {}
        if n_last:
            params["nult"] = n_last
        if date_start:
            params["dateStart"] = date_start
        if date_end:
            params["dateEnd"] = date_end

        return await self._request(f"ES/DATOS_SERIE/{series_id}", params if params else None)

    async def list_publications(self) -> list[dict[str, Any]]:
        """List all INE publications."""
        return await self._request("ES/PUBLICACIONES")

    async def get_publication(self, publication_id: str) -> dict[str, Any]:
        """Get details of a specific publication."""
        return await self._request(f"ES/PUBLICACION/{publication_id}")


# Global client instance
ine_client = INEClient()


def _handle_error(e: Exception, context: str = "ine_operation") -> str:
    """Format and log error message."""
    return handle_api_error(e, context=context, logger_name="ine")


def register_ine_tools(mcp):
    """Register INE tools with the MCP server."""

    @mcp.tool()
    async def ine_search(
        query: str | None = None,
        operation_id: str | None = None,
        page: int = 0,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> str:
        """Search INE statistical operations or list tables for an operation.

        INE is Spain's PRIMARY source for official statistics: demographics, economy,
        employment, prices (IPC/CPI), GDP, surveys, censuses, etc.

        Usage:
            - ine_search(query='empleo') → List operations matching 'empleo'
            - ine_search(operation_id='30308') → List tables for operation 30308 (EPA)

        Args:
            query: Search text to filter operations (e.g., 'empleo', 'poblacion', 'IPC').
            operation_id: If provided, lists tables for this operation instead of searching.
            page: Page number starting from 0.
            page_size: Results per page (default 50, max 100).

        Returns:
            JSON with operations (if query) or tables (if operation_id).

        Examples:
            ine_search(query='empleo') → Employment statistics (EPA, etc.)
            ine_search(query='IPC') → Consumer Price Index
            ine_search(operation_id='30308') → Tables for EPA operation
        """
        try:
            # If operation_id provided, list tables for that operation
            if operation_id:
                tables = await ine_client.list_tables(operation_id)
                output = {
                    "operation_id": operation_id,
                    "total_tables": len(tables),
                    "tables": [
                        {
                            "id": t.get("Id"),
                            "name": t.get("Nombre"),
                            "code": t.get("Codigo"),
                            "period": t.get("T3_Periodo"),
                            "publication": t.get("T3_Publicacion"),
                        }
                        for t in tables[:INE_MAX_TABLES]
                    ],
                }
                return json.dumps(output, ensure_ascii=False, indent=2)

            # Otherwise, search/list operations
            all_operations = await ine_client.list_operations()

            if query:
                query_lower = query.lower()
                all_operations = [
                    op for op in all_operations
                    if query_lower in op.get("Nombre", "").lower()
                ]

            page_size = min(max(1, page_size), 100)
            start = page * page_size
            end = start + page_size
            paginated = all_operations[start:end]

            output = {
                "query": query,
                "total_operations": len(all_operations),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(all_operations) + page_size - 1) // page_size if all_operations else 0,
                "operations": [
                    {
                        "id": op.get("Id"),
                        "code": op.get("Cod_IOE"),
                        "name": op.get("Nombre"),
                        "url": op.get("Url"),
                    }
                    for op in paginated
                ],
            }
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e, context="ine_search")

    @mcp.tool()
    async def ine_download(
        table_id: str,
        n_last: int = INE_DEFAULT_NLAST,
    ) -> str:
        """Download statistical data from an INE table.

        After finding a table with ine_search(operation_id=...), use this to get
        the actual statistics (numbers, percentages, indices).

        Args:
            table_id: Table ID from ine_search.
            n_last: Number of recent time periods to retrieve (default 10, max 100).

        Returns:
            JSON with statistical data including values, dates, units, and metadata.

        Example:
            ine_download('4247', n_last=12) → Last 12 periods of unemployment rate
        """
        try:
            n_last = min(max(1, n_last), 100)
            data = await ine_client.get_table_data(table_id, n_last=n_last)

            processed = []
            for item in data[:INE_MAX_DATA_RECORDS]:
                processed.append({
                    "name": item.get("Nombre"),
                    "value": item.get("Valor"),
                    "date": item.get("Fecha"),
                    "period": item.get("T3_Periodo"),
                    "unit": item.get("Unidad", {}).get("Nombre") if isinstance(item.get("Unidad"), dict) else item.get("Unidad"),
                })

            output = {
                "table_id": table_id,
                "n_last": n_last,
                "total_records": len(processed),
                "data": processed,
            }
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)
