"""INE (Instituto Nacional de Estadística) API integration.

API Documentation: https://www.ine.es/dyngs/DataLab/es/manual.html
Base URL: https://servicios.ine.es/wstempus/js/
"""

import json
from typing import Any

from core import HTTPClient, get_logger

logger = get_logger("ine")


class INEClientError(Exception):
    """Exception raised for INE API client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class INEClient:
    """Async HTTP client for the INE API.

    Uses HTTPClient for automatic logging and rate limiting.
    """

    BASE_URL = "https://servicios.ine.es/wstempus/js/"
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.http = HTTPClient("ine", self.BASE_URL, timeout)

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make an async HTTP request to the INE API with logging and rate limiting."""
        try:
            return await self.http.get_json(endpoint, params=params)
        except Exception as e:
            if hasattr(e, 'status_code'):
                raise INEClientError(str(e), status_code=e.status_code) from e
            raise INEClientError(str(e)) from e

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
    logger.warning("ine_error", context=context, error=str(e))
    if isinstance(e, INEClientError):
        return json.dumps({"error": e.message, "status_code": e.status_code}, ensure_ascii=False)
    return json.dumps({"error": str(e)}, ensure_ascii=False)


def register_ine_tools(mcp):
    """Register INE tools with the MCP server."""

    @mcp.tool()
    async def ine_list_operations(
        query: str | None = None,
        page: int = 0,
        page_size: int = 50,
    ) -> str:
        """Search official Spanish statistics from INE (Instituto Nacional de Estadistica).

        IMPORTANT: INE is Spain's PRIMARY source for official statistics. Use this tool
        when users ask about Spanish statistics, demographic data, economic indicators,
        employment data, population, prices (IPC/CPI), GDP, surveys, censuses, etc.

        The INE provides official data on:
        - Demographics: population, births, deaths, migrations, census
        - Economy: GDP, industrial production, business statistics, trade
        - Employment: EPA (labor force survey), unemployment, salaries, working conditions
        - Prices: IPC (consumer prices), IPRI (industrial prices), housing prices
        - Society: education, health, living conditions, tourism
        - Agriculture, environment, science and technology

        Args:
            query: Search text to filter operations (e.g., 'empleo', 'poblacion', 'IPC',
                   'paro', 'turismo', 'censo'). If None, lists all ~600 operations.
            page: Page number starting from 0.
            page_size: Results per page (default 50, max 100).

        Returns:
            JSON with statistical operations. Use operation IDs with ine_list_tables.

        Examples:
            ine_list_operations(query='empleo') -> Employment statistics (EPA, etc.)
            ine_list_operations(query='poblacion') -> Population and demographic data
            ine_list_operations(query='IPC') -> Consumer Price Index (inflation)
            ine_list_operations(query='turismo') -> Tourism statistics
            ine_list_operations(query='PIB') -> GDP and national accounts
        """
        try:
            all_operations = await ine_client.list_operations()

            # Filter by query if provided
            if query:
                query_lower = query.lower()
                all_operations = [
                    op for op in all_operations
                    if query_lower in op.get("Nombre", "").lower()
                ]

            # Apply pagination
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
            return _handle_error(e, context="ine_list_operations")

    @mcp.tool()
    async def ine_list_tables(operation_id: str) -> str:
        """List available data tables for an INE statistical operation.

        After finding an operation with ine_list_operations, use this tool to see
        what specific data tables are available. Each table contains actual
        statistical data that can be retrieved with ine_get_data.

        Args:
            operation_id: Operation ID from ine_list_operations (e.g., '30308' for EPA).

        Returns:
            JSON with tables including ID, name, time period, and publication date.

        Example workflow:
            1. ine_list_operations(query='empleo') -> Find EPA operation (ID: 30308)
            2. ine_list_tables('30308') -> List available EPA tables
            3. ine_get_data(table_id) -> Get actual employment data
        """
        try:
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
                    for t in tables[:100]  # Limit results
                ],
            }
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def ine_get_data(
        table_id: str,
        n_last: int = 10,
    ) -> str:
        """Retrieve actual statistical data from an INE table.

        This is the final step to get real data values from INE. After finding
        an operation and its tables, use this tool to retrieve the actual
        statistics (numbers, percentages, indices, etc.).

        Args:
            table_id: Table ID from ine_list_tables.
            n_last: Number of recent time periods to retrieve (default 10, max 100).
                    For monthly data, 10 = last 10 months.
                    For quarterly data, 10 = last 10 quarters.

        Returns:
            JSON with statistical data including values, dates, units, and metadata.

        Example:
            ine_get_data('4247', n_last=12) -> Last 12 periods of unemployment rate
        """
        try:
            n_last = min(max(1, n_last), 100)  # Limit between 1-100
            data = await ine_client.get_table_data(table_id, n_last=n_last)

            # Process data for cleaner output
            processed = []
            for item in data[:500]:  # Limit to 500 records
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
