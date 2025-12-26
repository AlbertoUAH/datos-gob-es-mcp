"""BOE (Boletín Oficial del Estado) API integration.

API Documentation: https://www.boe.es/datosabiertos/
Base URL: https://www.boe.es/datosabiertos/api/
"""

import json
from datetime import datetime
from typing import Any

from core import HTTPClient


class BOEClientError(Exception):
    """Exception raised for BOE API client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class BOEClient:
    """Async HTTP client for the BOE Open Data API.

    Uses HTTPClient for automatic logging and rate limiting.
    """

    BASE_URL = "https://www.boe.es/datosabiertos/api/"
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.http = HTTPClient("boe", self.BASE_URL, timeout)

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        response_format: str = "json"
    ) -> Any:
        """Make an async HTTP request to the BOE API with logging and rate limiting."""
        # BOE API uses format in URL path
        url = endpoint
        if not url.endswith("/"):
            url += "/"
        url += f"?formato={response_format}"

        try:
            response = await self.http.get(url, params=params)

            if response_format == "json":
                return response.json()
            return response.text

        except Exception as e:
            if hasattr(e, 'status_code'):
                raise BOEClientError(str(e), status_code=e.status_code) from e
            raise BOEClientError(str(e)) from e

    async def get_summary(self, date: str) -> dict[str, Any]:
        """Get BOE summary for a specific date.

        Args:
            date: Date in format YYYYMMDD
        """
        return await self._request(f"boe/sumario/{date}")

    async def get_document(self, document_id: str) -> dict[str, Any]:
        """Get a specific BOE document.

        Args:
            document_id: Document identifier (e.g., 'BOE-A-2024-12345')
        """
        return await self._request(f"boe/documento/{document_id}")

    async def search(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        department: str | None = None,
        page: int = 1,
    ) -> dict[str, Any]:
        """Search BOE documents.

        Args:
            query: Search query
            date_from: Start date YYYYMMDD
            date_to: End date YYYYMMDD
            department: Department/ministry filter
            page: Page number
        """
        # BOE search is done via the BORME/BOE API endpoint
        # Construct search URL
        params = {"texto": query, "pagina": page}
        if date_from:
            params["fpu"] = date_from
        if date_to:
            params["fpf"] = date_to
        if department:
            params["departamento"] = department

        return await self._request("boe/dias", params)

    async def get_recent_summaries(self, days: int = 7) -> list[dict[str, Any]]:
        """Get BOE summaries for recent days.

        Args:
            days: Number of days to fetch (default 7)
        """
        summaries = []
        today = datetime.now()

        for i in range(days):
            date = today.replace(day=today.day - i)
            date_str = date.strftime("%Y%m%d")
            try:
                summary = await self.get_summary(date_str)
                if summary:
                    summaries.append(summary)
            except BOEClientError:
                # Skip days without BOE (weekends, holidays)
                continue

        return summaries

    async def get_borme_summary(self, date: str) -> dict[str, Any]:
        """Get BORME (Boletín Oficial del Registro Mercantil) summary.

        Args:
            date: Date in format YYYYMMDD
        """
        return await self._request(f"borme/sumario/{date}")

    async def get_borme_document(self, document_id: str) -> dict[str, Any]:
        """Get a specific BORME document.

        Args:
            document_id: Document identifier
        """
        return await self._request(f"borme/documento/{document_id}")


# Global client instance
boe_client = BOEClient()


def _handle_error(e: Exception) -> str:
    """Format error message."""
    if isinstance(e, BOEClientError):
        return json.dumps({"error": e.message, "status_code": e.status_code}, ensure_ascii=False)
    return json.dumps({"error": str(e)}, ensure_ascii=False)


def _format_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Format BOE summary for cleaner output."""
    if not summary:
        return {"error": "No summary available"}

    data = summary.get("data", {})
    sumario = data.get("sumario", {})
    meta = sumario.get("metadatos", {})
    diario = sumario.get("diario", [])

    # Extract sections
    secciones = []
    for seccion in diario:
        seccion_info = {
            "nombre": seccion.get("sumario_nombre"),
            "departamentos": [],
        }

        for dept in seccion.get("seccion", []):
            dept_info = {
                "nombre": dept.get("departamento"),
                "epigrafes": [],
            }

            for epigrafe in dept.get("departamento_epigrafe", []):
                for item in epigrafe.get("item", []):
                    dept_info["epigrafes"].append({
                        "id": item.get("identificador"),
                        "titulo": item.get("titulo"),
                        "url_pdf": item.get("url_pdf", {}).get("texto") if isinstance(item.get("url_pdf"), dict) else item.get("url_pdf"),
                    })

            if dept_info["epigrafes"]:
                seccion_info["departamentos"].append(dept_info)

        if seccion_info["departamentos"]:
            secciones.append(seccion_info)

    return {
        "fecha": meta.get("fecha_publicacion"),
        "numero": meta.get("pub_numero"),
        "total_paginas": meta.get("numero_paginas"),
        "secciones": secciones,
    }


def _format_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Format BOE document for cleaner output."""
    if not doc:
        return {"error": "Document not found"}

    data = doc.get("data", {})
    documento = data.get("documento", {})
    meta = documento.get("metadatos", {})
    analisis = documento.get("analisis", {})

    return {
        "id": meta.get("identificador"),
        "titulo": meta.get("titulo"),
        "fecha_publicacion": meta.get("fecha_publicacion"),
        "departamento": meta.get("departamento", {}).get("texto") if isinstance(meta.get("departamento"), dict) else meta.get("departamento"),
        "rango": meta.get("rango", {}).get("texto") if isinstance(meta.get("rango"), dict) else meta.get("rango"),
        "seccion": meta.get("seccion"),
        "origen_legislativo": meta.get("origen_legislativo"),
        "url_pdf": meta.get("url_pdf", {}).get("texto") if isinstance(meta.get("url_pdf"), dict) else meta.get("url_pdf"),
        "url_html": meta.get("url_html", {}).get("texto") if isinstance(meta.get("url_html"), dict) else meta.get("url_html"),
        "materias": analisis.get("materias", []) if analisis else [],
        "notas": analisis.get("notas", []) if analisis else [],
    }


def register_boe_tools(mcp):
    """Register BOE tools with the MCP server."""

    @mcp.tool()
    async def boe_get_summary(date: str) -> str:
        """Get the BOE (Official State Gazette) summary for a specific date.

        Retrieve the complete index of all documents published in the BOE
        for a given date. Includes laws, royal decrees, resolutions, and
        other official publications.

        Args:
            date: Date in format YYYYMMDD (e.g., '20241226' for December 26, 2024).
                Note: BOE is not published on weekends or public holidays.

        Returns:
            JSON with BOE summary including sections, departments, and document titles.
        """
        try:
            data = await boe_client.get_summary(date)
            formatted = _format_summary(data)
            return json.dumps(formatted, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def boe_get_document(document_id: str) -> str:
        """Get a specific BOE document by its identifier.

        Retrieve full metadata and analysis for a BOE document.
        Includes title, department, legal classification, and related documents.

        Args:
            document_id: BOE document identifier (e.g., 'BOE-A-2024-12345').
                Format: BOE-{section}-{year}-{number}

        Returns:
            JSON with document metadata, PDF/HTML URLs, and legal analysis.
        """
        try:
            data = await boe_client.get_document(document_id)
            formatted = _format_document(data)
            return json.dumps(formatted, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def boe_search(
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        """Search for documents in the BOE.

        Search the Official State Gazette for documents matching a query.
        Can filter by date range.

        Args:
            query: Search text (e.g., 'subvenciones', 'real decreto', 'educación').
            date_from: Optional start date in format YYYYMMDD.
            date_to: Optional end date in format YYYYMMDD.

        Returns:
            JSON with matching documents including IDs and titles.
        """
        try:
            # For search, we'll get recent summaries and filter
            # since BOE doesn't have a direct search API in the same format
            days_to_search = 30
            if date_from and date_to:
                # Calculate days between dates
                from datetime import datetime
                d1 = datetime.strptime(date_from, "%Y%m%d")
                d2 = datetime.strptime(date_to, "%Y%m%d")
                days_to_search = min((d2 - d1).days + 1, 90)

            # Get summaries and search within them
            results = []
            query_lower = query.lower()

            # Get today's date or date_to as starting point
            if date_to:
                current = datetime.strptime(date_to, "%Y%m%d")
            else:
                current = datetime.now()

            for i in range(days_to_search):
                check_date = current.replace(day=current.day - i) if i > 0 else current
                try:
                    # Handle month/year boundaries
                    from datetime import timedelta
                    check_date = current - timedelta(days=i)
                    date_str = check_date.strftime("%Y%m%d")

                    if date_from and date_str < date_from:
                        break

                    summary = await boe_client.get_summary(date_str)
                    formatted = _format_summary(summary)

                    # Search in sections
                    for seccion in formatted.get("secciones", []):
                        for dept in seccion.get("departamentos", []):
                            for epigrafe in dept.get("epigrafes", []):
                                titulo = epigrafe.get("titulo", "")
                                if query_lower in titulo.lower():
                                    results.append({
                                        "id": epigrafe.get("id"),
                                        "titulo": titulo,
                                        "fecha": formatted.get("fecha"),
                                        "seccion": seccion.get("nombre"),
                                        "departamento": dept.get("nombre"),
                                    })

                except BOEClientError:
                    continue

                if len(results) >= 50:  # Limit results
                    break

            output = {
                "query": query,
                "date_from": date_from,
                "date_to": date_to,
                "total_results": len(results),
                "documents": results,
            }
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def boe_get_today() -> str:
        """Get today's BOE summary.

        Retrieve the summary of today's Official State Gazette.
        If today's BOE is not available (weekend/holiday), returns
        the most recent available BOE.

        Returns:
            JSON with today's (or most recent) BOE summary.
        """
        try:
            # Try today and previous days until we find a BOE
            from datetime import datetime, timedelta

            for i in range(7):  # Try up to 7 days back
                date = datetime.now() - timedelta(days=i)
                date_str = date.strftime("%Y%m%d")
                try:
                    data = await boe_client.get_summary(date_str)
                    if data:
                        formatted = _format_summary(data)
                        formatted["note"] = f"BOE del {date.strftime('%d/%m/%Y')}"
                        return json.dumps(formatted, ensure_ascii=False, indent=2)
                except BOEClientError:
                    continue

            return json.dumps({"error": "No BOE available in the last 7 days"}, ensure_ascii=False)
        except Exception as e:
            return _handle_error(e)
