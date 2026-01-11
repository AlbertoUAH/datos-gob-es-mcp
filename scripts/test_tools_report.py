#!/usr/bin/env python3
"""
MCP Tools Integration Test Suite

This script runs comprehensive tests against all MCP tools and generates
a detailed markdown report at docs/test_report.md.

Usage:
    python scripts/test_tools_report.py
"""

import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the internal functions, not the MCP tool decorators
# These are the actual async functions that do the work
from integrations.ine import ine_client
from integrations.aemet import aemet_client
from integrations.boe import boe_client


@dataclass
class TestResult:
    """Result of a single test execution."""

    name: str
    tool: str
    category: str
    passed: bool
    duration_ms: float
    input_params: dict[str, Any]
    output_summary: str
    output_full: str | None = None
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ToolTester:
    """Executes tests against MCP tools."""

    def __init__(self):
        self.results: list[TestResult] = []

    async def run_test(
        self,
        name: str,
        tool: str,
        category: str,
        func,
        params: dict[str, Any],
        validate_fn=None,
        expect_error: bool = False,
        expect_error_response: bool = False,
    ) -> TestResult:
        """Execute a single test and record the result.

        Args:
            expect_error: If True, the test passes when an exception is raised
                         (for testing error handling of non-existent resources)
            expect_error_response: If True, the test passes when "error" is in the response
                                   (for testing error responses from APIs)
        """
        start = time.perf_counter()

        try:
            # Execute the tool
            if asyncio.iscoroutinefunction(func):
                result = await func(**params)
            else:
                result = func(**params)

            duration = (time.perf_counter() - start) * 1000

            # If we expected an error but didn't get one
            if expect_error:
                test_result = TestResult(
                    name=name,
                    tool=tool,
                    category=category,
                    passed=False,
                    duration_ms=duration,
                    input_params=params,
                    output_summary="Expected error but got success",
                    output_full=str(result)[:5000],
                )
                self.results.append(test_result)
                status = "❌"
                print(f"  {status} {name} ({test_result.duration_ms:.0f}ms)")
                return test_result

            # Parse result if it's JSON string
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    output_full = json.dumps(parsed, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    parsed = result
                    output_full = result
            else:
                parsed = result
                output_full = str(result)

            # Generate summary
            if isinstance(parsed, dict):
                if "error" in parsed:
                    summary = f"Error: {parsed['error'][:100]}"
                    # If we expected an error response, this is a pass
                    passed = expect_error_response
                    if expect_error_response:
                        summary = f"Expected error response: {parsed['error'][:100]}"
                else:
                    keys = list(parsed.keys())[:5]
                    summary = f"Keys: {keys}"
                    passed = True
            elif isinstance(parsed, list):
                summary = f"List with {len(parsed)} items"
                passed = len(parsed) > 0 or validate_fn is None
            else:
                summary = str(parsed)[:100]
                passed = True

            # Custom validation
            if validate_fn and passed:
                try:
                    passed = validate_fn(parsed)
                    if not passed:
                        summary += " (validation failed)"
                except Exception as e:
                    passed = False
                    summary += f" (validation error: {e})"

            test_result = TestResult(
                name=name,
                tool=tool,
                category=category,
                passed=passed,
                duration_ms=duration,
                input_params=params,
                output_summary=summary,
                output_full=output_full[:5000] if output_full else None,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000

            # If we expected an error, this is a pass
            if expect_error:
                test_result = TestResult(
                    name=name,
                    tool=tool,
                    category=category,
                    passed=True,
                    duration_ms=duration,
                    input_params=params,
                    output_summary=f"Expected error: {type(e).__name__}",
                    output_full=f"{type(e).__name__}: {str(e)}"
                )
            else:
                test_result = TestResult(
                    name=name,
                    tool=tool,
                    category=category,
                    passed=False,
                    duration_ms=duration,
                    input_params=params,
                    output_summary=f"Exception: {type(e).__name__}",
                    error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                )

        self.results.append(test_result)
        status = "✅" if test_result.passed else "❌"
        print(f"  {status} {name} ({test_result.duration_ms:.0f}ms)")
        return test_result

    # =========================================================================
    # datos.gob.es Tests
    # =========================================================================

    async def test_search(self):
        """Test search tool with various parameters."""
        print("\n📦 Testing: search (datos.gob.es)")

        # Test 1: Search by title (using normalized term without accent)
        await self.run_test(
            name="Búsqueda por título (poblacion)",
            tool="search",
            category="datos.gob.es",
            func=self._search_by_title,
            params={"title": "poblacion", "max_results": 5},
            validate_fn=lambda r: len(r.get("datasets", [])) > 0
        )

        # Test 2: Search by theme
        await self.run_test(
            name="Búsqueda por tema (economía)",
            tool="search",
            category="datos.gob.es",
            func=self._search_by_theme,
            params={"theme": "economia", "max_results": 5},
            validate_fn=lambda r: len(r.get("datasets", [])) > 0
        )

        # Test 3: Search by format
        await self.run_test(
            name="Búsqueda por formato CSV",
            tool="search",
            category="datos.gob.es",
            func=self._search_by_format,
            params={"format": "csv", "max_results": 5},
            validate_fn=lambda r: len(r.get("datasets", [])) > 0
        )

        # Test 4: Search by keyword (using normalized term without accent)
        await self.run_test(
            name="Búsqueda por keyword (estadistica)",
            tool="search",
            category="datos.gob.es",
            func=self._search_by_keyword,
            params={"keyword": "estadistica", "max_results": 5},
            validate_fn=lambda r: len(r.get("datasets", [])) > 0
        )

    async def _search_by_title(self, title, max_results=50):
        """Search datos.gob.es by title."""
        from core import DATOS_GOB_BASE_URL, HTTPClient
        http = HTTPClient("datos.gob.es", DATOS_GOB_BASE_URL)
        response = await http.get_json(f"catalog/dataset/title/{title}.json", params={"_pageSize": max_results})
        items = response.get("result", {}).get("items", [])
        return {
            "total": response.get("result", {}).get("totalResults", 0),
            "datasets": [{"id": self._extract_id(i), "title": self._extract_title(i)} for i in items[:max_results]]
        }

    async def _search_by_theme(self, theme, max_results=50):
        """Search datos.gob.es by theme."""
        from core import DATOS_GOB_BASE_URL, HTTPClient
        http = HTTPClient("datos.gob.es", DATOS_GOB_BASE_URL)
        response = await http.get_json(f"catalog/dataset/theme/{theme}.json", params={"_pageSize": max_results})
        items = response.get("result", {}).get("items", [])
        return {
            "total": response.get("result", {}).get("totalResults", 0),
            "datasets": [{"id": self._extract_id(i), "title": self._extract_title(i)} for i in items[:max_results]]
        }

    async def _search_by_format(self, format, max_results=50):
        """Search datos.gob.es by format."""
        from core import DATOS_GOB_BASE_URL, HTTPClient
        http = HTTPClient("datos.gob.es", DATOS_GOB_BASE_URL)
        response = await http.get_json(f"catalog/dataset/format/{format}.json", params={"_pageSize": max_results})
        items = response.get("result", {}).get("items", [])
        return {
            "total": response.get("result", {}).get("totalResults", 0),
            "datasets": [{"id": self._extract_id(i), "title": self._extract_title(i)} for i in items[:max_results]]
        }

    async def _search_by_keyword(self, keyword, max_results=50):
        """Search datos.gob.es by keyword."""
        from core import DATOS_GOB_BASE_URL, HTTPClient
        http = HTTPClient("datos.gob.es", DATOS_GOB_BASE_URL)
        response = await http.get_json(f"catalog/dataset/keyword/{keyword}.json", params={"_pageSize": max_results})
        items = response.get("result", {}).get("items", [])
        return {
            "total": response.get("result", {}).get("totalResults", 0),
            "datasets": [{"id": self._extract_id(i), "title": self._extract_title(i)} for i in items[:max_results]]
        }

    def _extract_id(self, item):
        """Extract dataset ID from API item."""
        about = item.get("_about", "")
        return about.split("/")[-1] if about else None

    def _extract_title(self, item):
        """Extract title from API item."""
        title = item.get("title", {})
        if isinstance(title, dict):
            return title.get("_value", "")
        elif isinstance(title, list) and title:
            return title[0].get("_value", "") if isinstance(title[0], dict) else str(title[0])
        return str(title)

    async def test_get(self):
        """Test get tool with various parameters."""
        print("\n📦 Testing: get (datos.gob.es)")

        # Test 1: Get metadata only (using valid dataset ID)
        await self.run_test(
            name="Obtener metadatos de dataset INE",
            tool="get",
            category="datos.gob.es",
            func=self._get_dataset,
            params={"dataset_id": "ea0010587-valor-anadido-bruto-cneag-identificador-api-67197"},
            validate_fn=lambda r: "title" in r or "_about" in r
        )

        # Test 2: Get another dataset (using valid dataset ID)
        await self.run_test(
            name="Obtener dataset medio ambiente",
            tool="get",
            category="datos.gob.es",
            func=self._get_dataset,
            params={"dataset_id": "e05068001-mapas-estrategicos-de-ruido"},
        )

        # Test 3: Get non-existent dataset (expected error response)
        await self.run_test(
            name="Dataset inexistente (error esperado)",
            tool="get",
            category="datos.gob.es",
            func=self._get_dataset,
            params={"dataset_id": "dataset-que-no-existe-12345"},
            expect_error_response=True,
        )

    async def _get_dataset(self, dataset_id):
        """Get dataset metadata from datos.gob.es."""
        from core import DATOS_GOB_BASE_URL, HTTPClient
        http = HTTPClient("datos.gob.es", DATOS_GOB_BASE_URL)
        try:
            response = await http.get_json(f"catalog/dataset/{dataset_id}.json")
            items = response.get("result", {}).get("items", [])
            if items:
                item = items[0]
                # Extract publisher safely (can be dict or string)
                publisher = item.get("publisher", "")
                if isinstance(publisher, dict):
                    publisher = publisher.get("_about", "")
                # Extract distributions safely (can be list or dict)
                distribution = item.get("distribution", [])
                if isinstance(distribution, dict):
                    distribution = [distribution]
                return {
                    "id": self._extract_id(item),
                    "title": self._extract_title(item),
                    "description": self._extract_desc(item),
                    "publisher": publisher if isinstance(publisher, str) else str(publisher),
                    "distributions": len(distribution) if isinstance(distribution, list) else 1,
                }
            return {"error": "Dataset not found", "total": 0}
        except Exception as e:
            return {"error": str(e), "total": 0}

    def _extract_desc(self, item):
        """Extract description from API item."""
        desc = item.get("description", {})
        if isinstance(desc, dict):
            return desc.get("_value", "")[:200]
        elif isinstance(desc, list) and desc:
            return desc[0].get("_value", "")[:200] if isinstance(desc[0], dict) else str(desc[0])[:200]
        return str(desc)[:200]

    # =========================================================================
    # INE Tests
    # =========================================================================

    async def test_ine_search(self):
        """Test INE search tool."""
        print("\n📊 Testing: ine_search (INE)")

        # Test 1: Search operations by query
        await self.run_test(
            name="Buscar operaciones (empleo)",
            tool="ine_search",
            category="INE",
            func=self._ine_search_wrapper,
            params={"query": "empleo"},
            validate_fn=lambda r: r.get("total_operations", 0) > 0
        )

        # Test 2: Search operations (population)
        await self.run_test(
            name="Buscar operaciones (población)",
            tool="ine_search",
            category="INE",
            func=self._ine_search_wrapper,
            params={"query": "poblacion"},
            validate_fn=lambda r: r.get("total_operations", 0) > 0
        )

        # Test 3: List all operations
        await self.run_test(
            name="Listar todas las operaciones",
            tool="ine_search",
            category="INE",
            func=self._ine_search_wrapper,
            params={"page_size": 10},
            validate_fn=lambda r: r.get("total_operations", 0) > 0
        )

        # Test 4: Get tables for valid operation (IPC - Id: 25)
        await self.run_test(
            name="Obtener tablas de operación (IPC)",
            tool="ine_search",
            category="INE",
            func=self._ine_search_wrapper,
            params={"operation_id": "25"},
            validate_fn=lambda r: r.get("total_tables", 0) > 0
        )

        # Test 5: Get tables for operation without tables
        await self.run_test(
            name="Operación sin tablas (respuesta vacía)",
            tool="ine_search",
            category="INE",
            func=self._ine_search_wrapper,
            params={"operation_id": "99999"},
            validate_fn=lambda r: r.get("total_tables", 0) == 0
        )

    async def _ine_search_wrapper(self, query=None, operation_id=None, page=0, page_size=50):
        """Wrapper for INE search that returns parsed JSON.

        Uses synonym expansion to match the actual MCP tool behavior.
        """
        from integrations.ine import _expand_query_with_synonyms

        if operation_id:
            tables = await ine_client.list_tables(operation_id)
            return {
                "operation_id": operation_id,
                "total_tables": len(tables),
                "tables": tables[:10]
            }
        else:
            operations = await ine_client.list_operations()
            if query:
                # Use synonym expansion like the actual MCP tool
                search_terms = _expand_query_with_synonyms(query)
                matching_ops = []
                seen_ids = set()
                for op in operations:
                    op_name = op.get("Nombre", "").lower()
                    op_id = op.get("Id")
                    if op_id in seen_ids:
                        continue
                    for term in search_terms:
                        if term in op_name:
                            matching_ops.append(op)
                            seen_ids.add(op_id)
                            break
                operations = matching_ops
            return {
                "query": query,
                "total_operations": len(operations),
                "operations": operations[page * page_size:(page + 1) * page_size]
            }

    async def test_ine_download(self):
        """Test INE download tool."""
        print("\n📊 Testing: ine_download (INE)")

        # Test 1: Download valid table data (IPC table)
        await self.run_test(
            name="Descargar datos de tabla IPC",
            tool="ine_download",
            category="INE",
            func=self._ine_download_wrapper,
            params={"table_id": "50902", "n_last": 5},
            validate_fn=lambda r: len(r.get("data", [])) > 0
        )

        # Test 2: Download with more periods
        await self.run_test(
            name="Descargar más períodos",
            tool="ine_download",
            category="INE",
            func=self._ine_download_wrapper,
            params={"table_id": "50902", "n_last": 12},
        )

        # Test 3: Invalid table (expected to fail)
        await self.run_test(
            name="Tabla inexistente (error esperado)",
            tool="ine_download",
            category="INE",
            func=self._ine_download_wrapper,
            params={"table_id": "99999999", "n_last": 5},
            expect_error=True,
        )

    async def _ine_download_wrapper(self, table_id, n_last=10):
        """Wrapper for INE download."""
        data = await ine_client.get_table_data(table_id, n_last=n_last)
        return {
            "table_id": table_id,
            "n_last": n_last,
            "total_records": len(data) if data else 0,
            "data": data[:20] if data else []
        }

    # =========================================================================
    # AEMET Tests
    # =========================================================================

    async def test_aemet_forecast(self):
        """Test AEMET forecast tool."""
        print("\n🌤️ Testing: aemet_get_forecast (AEMET)")

        # Test 1: Forecast for Madrid
        await self.run_test(
            name="Pronóstico Madrid (28079)",
            tool="aemet_get_forecast",
            category="AEMET",
            func=self._aemet_forecast_wrapper,
            params={"municipality_code": "28079"},
            validate_fn=lambda r: len(r) > 0 and "nombre" in r[0]
        )

        # Test 2: Forecast for Barcelona
        await self.run_test(
            name="Pronóstico Barcelona (08019)",
            tool="aemet_get_forecast",
            category="AEMET",
            func=self._aemet_forecast_wrapper,
            params={"municipality_code": "08019"},
        )

        # Test 3: Invalid municipality (expected to fail)
        await self.run_test(
            name="Municipio inexistente (error esperado)",
            tool="aemet_get_forecast",
            category="AEMET",
            func=self._aemet_forecast_wrapper,
            params={"municipality_code": "99999"},
            expect_error=True,
        )

    async def _aemet_forecast_wrapper(self, municipality_code):
        """Wrapper for AEMET forecast."""
        return await aemet_client.get_forecast_daily(municipality_code)

    async def test_aemet_observations(self):
        """Test AEMET observations tool."""
        print("\n🌤️ Testing: aemet_get_observations (AEMET)")

        # Test 1: All stations
        await self.run_test(
            name="Observaciones todas las estaciones",
            tool="aemet_get_observations",
            category="AEMET",
            func=self._aemet_observations_wrapper,
            params={},
            validate_fn=lambda r: len(r) > 0
        )

        # Test 2: Specific station (Madrid-Retiro)
        await self.run_test(
            name="Observaciones Madrid-Retiro (3129)",
            tool="aemet_get_observations",
            category="AEMET",
            func=self._aemet_observations_wrapper,
            params={"station_id": "3129"},
        )

    async def _aemet_observations_wrapper(self, station_id=None):
        """Wrapper for AEMET observations."""
        return await aemet_client.get_observations(station_id)

    async def test_aemet_locations(self):
        """Test AEMET locations tool."""
        print("\n🌤️ Testing: aemet_list_locations (AEMET)")

        # Test 1: List municipalities
        await self.run_test(
            name="Listar municipios",
            tool="aemet_list_locations",
            category="AEMET",
            func=self._aemet_municipalities_wrapper,
            params={},
            validate_fn=lambda r: len(r) > 0
        )

        # Test 2: List stations
        await self.run_test(
            name="Listar estaciones",
            tool="aemet_list_locations",
            category="AEMET",
            func=self._aemet_stations_wrapper,
            params={},
            validate_fn=lambda r: len(r) > 0
        )

    async def _aemet_municipalities_wrapper(self):
        """Wrapper for AEMET municipalities."""
        return await aemet_client.get_municipalities()

    async def _aemet_stations_wrapper(self):
        """Wrapper for AEMET stations."""
        return await aemet_client.get_stations()

    # =========================================================================
    # BOE Tests
    # =========================================================================

    async def test_boe_summary(self):
        """Test BOE summary tool."""
        print("\n📜 Testing: boe_get_summary (BOE)")

        # Test 1: Get most recent BOE
        await self.run_test(
            name="BOE más reciente",
            tool="boe_get_summary",
            category="BOE",
            func=self._boe_summary_wrapper,
            params={},
            validate_fn=lambda r: "fecha" in r and r.get("fecha") is not None
        )

        # Test 2: Get specific date (recent working day)
        from datetime import timedelta
        recent_date = datetime.now() - timedelta(days=3)
        date_str = recent_date.strftime("%Y%m%d")
        await self.run_test(
            name=f"BOE fecha específica ({date_str})",
            tool="boe_get_summary",
            category="BOE",
            func=self._boe_summary_wrapper,
            params={"date": date_str},
        )

    async def _boe_summary_wrapper(self, date=None):
        """Wrapper for BOE summary."""
        from integrations.boe import _format_summary

        if date is None:
            # Get most recent
            from datetime import timedelta
            today = datetime.now()
            for i in range(7):
                check_date = today - timedelta(days=i)
                date_str = check_date.strftime("%Y%m%d")
                try:
                    data = await boe_client.get_summary(date_str)
                    if data:
                        result = _format_summary(data)
                        result["note"] = f"BOE del {check_date.strftime('%d/%m/%Y')}"
                        return result
                except Exception:
                    continue
            return {"error": "No BOE found in last 7 days"}
        else:
            data = await boe_client.get_summary(date)
            return _format_summary(data)

    async def test_boe_document(self):
        """Test BOE document tool."""
        print("\n📜 Testing: boe_get_document (BOE)")

        # First get a valid document ID from recent BOE
        try:
            from datetime import timedelta
            today = datetime.now()
            doc_id = None
            for i in range(7):
                check_date = today - timedelta(days=i)
                date_str = check_date.strftime("%Y%m%d")
                try:
                    data = await boe_client.get_summary(date_str)
                    if data:
                        # Extract first document ID (with type checks for variable structure)
                        sumario = data.get("data", {}).get("sumario", {})
                        diario_list = sumario.get("diario", [])
                        if isinstance(diario_list, dict):
                            diario_list = [diario_list]
                        for diario in diario_list:
                            if not isinstance(diario, dict):
                                continue
                            seccion_list = diario.get("seccion", [])
                            if isinstance(seccion_list, dict):
                                seccion_list = [seccion_list]
                            for seccion in seccion_list:
                                if not isinstance(seccion, dict):
                                    continue
                                dept_list = seccion.get("departamento", [])
                                if isinstance(dept_list, dict):
                                    dept_list = [dept_list]
                                for dept in dept_list:
                                    if not isinstance(dept, dict):
                                        continue
                                    epigrafe_list = dept.get("epigrafe", [])
                                    if isinstance(epigrafe_list, dict):
                                        epigrafe_list = [epigrafe_list]
                                    for epigrafe in epigrafe_list:
                                        if not isinstance(epigrafe, dict):
                                            continue
                                        items = epigrafe.get("item", [])
                                        if isinstance(items, dict):
                                            items = [items]
                                        for item in items:
                                            if not isinstance(item, dict):
                                                continue
                                            doc_id = item.get("identificador")
                                            if doc_id:
                                                break
                                        if doc_id:
                                            break
                                    if doc_id:
                                        break
                                if doc_id:
                                    break
                            if doc_id:
                                break
                        if doc_id:
                            break
                except Exception:
                    continue
        except Exception:
            doc_id = "BOE-A-2024-1"

        # Test 1: Get valid document
        if doc_id:
            await self.run_test(
                name=f"Obtener documento ({doc_id})",
                tool="boe_get_document",
                category="BOE",
                func=self._boe_document_wrapper,
                params={"document_id": doc_id},
                validate_fn=lambda r: "id" in r or "titulo" in r or "error" not in r
            )

        # Test 2: Invalid document (expected to fail)
        await self.run_test(
            name="Documento inexistente (error esperado)",
            tool="boe_get_document",
            category="BOE",
            func=self._boe_document_wrapper,
            params={"document_id": "BOE-X-9999-99999"},
            expect_error=True,
        )

    async def _boe_document_wrapper(self, document_id):
        """Wrapper for BOE document."""
        from integrations.boe import _format_document
        data = await boe_client.get_document(document_id)
        return _format_document(data)

    async def test_boe_search(self):
        """Test BOE search tool."""
        print("\n📜 Testing: boe_search (BOE)")

        # Test 1: Basic search
        await self.run_test(
            name="Búsqueda básica (subvenciones)",
            tool="boe_search",
            category="BOE",
            func=self._boe_search_wrapper,
            params={"query": "subvenciones"},
        )

        # Test 2: Search with date range
        await self.run_test(
            name="Búsqueda con fechas",
            tool="boe_search",
            category="BOE",
            func=self._boe_search_wrapper,
            params={
                "query": "educación",
                "date_from": "20240101",
                "date_to": "20240331"
            },
        )

    async def _boe_search_wrapper(self, query, date_from=None, date_to=None):
        """Wrapper for BOE search."""
        # BOE search via summaries
        from datetime import timedelta
        results = []

        if date_from and date_to:
            start = datetime.strptime(date_from, "%Y%m%d")
            end = datetime.strptime(date_to, "%Y%m%d")
        else:
            end = datetime.now()
            start = end - timedelta(days=7)

        query_lower = query.lower()
        current = start
        while current <= end and len(results) < 10:
            date_str = current.strftime("%Y%m%d")
            try:
                data = await boe_client.get_summary(date_str)
                if data:
                    sumario = data.get("data", {}).get("sumario", {})
                    for diario in sumario.get("diario", []):
                        for seccion in diario.get("seccion", []):
                            for dept in seccion.get("departamento", []):
                                for epigrafe in dept.get("epigrafe", []):
                                    items = epigrafe.get("item", [])
                                    if isinstance(items, dict):
                                        items = [items]
                                    for item in items:
                                        titulo = item.get("titulo", "").lower()
                                        if query_lower in titulo:
                                            results.append({
                                                "id": item.get("identificador"),
                                                "titulo": item.get("titulo"),
                                                "fecha": date_str,
                                            })
            except Exception:
                pass
            current += timedelta(days=1)

        return {
            "query": query,
            "total_results": len(results),
            "results": results[:20]
        }

    # =========================================================================
    # Run All Tests
    # =========================================================================

    async def run_all_tests(self):
        """Execute all tests."""
        print("=" * 60)
        print("🧪 MCP Tools Integration Test Suite")
        print("=" * 60)

        start_time = time.perf_counter()

        # datos.gob.es
        await self.test_search()
        await self.test_get()

        # INE
        await self.test_ine_search()
        await self.test_ine_download()

        # AEMET (with delays to avoid rate limiting)
        print("\n⏳ Esperando 60s para evitar rate limit de AEMET...")
        await asyncio.sleep(60)
        await self.test_aemet_forecast()
        print("\n⏳ Esperando 60s para evitar rate limit de AEMET...")
        await asyncio.sleep(60)
        await self.test_aemet_observations()
        print("\n⏳ Esperando 60s para evitar rate limit de AEMET...")
        await asyncio.sleep(60)
        await self.test_aemet_locations()

        # BOE
        await self.test_boe_summary()
        await self.test_boe_document()
        await self.test_boe_search()

        total_time = (time.perf_counter() - start_time) * 1000

        print("\n" + "=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        print(f"✅ Passed: {passed} | ❌ Failed: {failed} | Total: {len(self.results)}")
        print(f"⏱️  Total time: {total_time:.0f}ms")
        print("=" * 60)

        return self.results


class MarkdownReporter:
    """Generates markdown report from test results."""

    def __init__(self, results: list[TestResult]):
        self.results = results
        self.timestamp = datetime.now()

    def generate(self) -> str:
        """Generate complete markdown report."""
        sections = [
            self._generate_header(),
            self._generate_summary(),
            self._generate_results_by_category(),
            self._generate_statistics(),
            self._generate_technical_details(),
        ]
        return "\n\n".join(sections)

    def _generate_header(self) -> str:
        """Generate report header."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        status = "✅ All Passed" if failed == 0 else f"⚠️ {failed} Failed"

        return f"""# 🧪 MCP Tools Test Report

> **Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
> **Status:** {status}
> **Total Tests:** {len(self.results)}"""

    def _generate_summary(self) -> str:
        """Generate executive summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_time = sum(r.duration_ms for r in self.results)
        avg_time = total_time / len(self.results) if self.results else 0

        # Group by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"passed": 0, "failed": 0, "time": 0}
            categories[r.category]["time"] += r.duration_ms
            if r.passed:
                categories[r.category]["passed"] += 1
            else:
                categories[r.category]["failed"] += 1

        summary = f"""## 📊 Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Tests Pasados | {passed} ✅ |
| Tests Fallidos | {failed} ❌ |
| Tiempo Total | {total_time:.0f} ms |
| Tiempo Promedio | {avg_time:.0f} ms |

### Por Categoría

| Categoría | Pasados | Fallidos | Tiempo (ms) |
|-----------|---------|----------|-------------|"""

        for cat, stats in categories.items():
            emoji = "✅" if stats["failed"] == 0 else "⚠️"
            summary += f"\n| {emoji} {cat} | {stats['passed']} | {stats['failed']} | {stats['time']:.0f} |"

        return summary

    def _generate_results_by_category(self) -> str:
        """Generate detailed results by category."""
        sections = ["## 🔧 Resultados Detallados"]

        # Group by category and tool
        by_category = {}
        for r in self.results:
            if r.category not in by_category:
                by_category[r.category] = {}
            if r.tool not in by_category[r.category]:
                by_category[r.category][r.tool] = []
            by_category[r.category][r.tool].append(r)

        category_emojis = {
            "datos.gob.es": "📦",
            "INE": "📊",
            "AEMET": "🌤️",
            "BOE": "📜"
        }

        for category, tools in by_category.items():
            emoji = category_emojis.get(category, "🔧")
            sections.append(f"\n### {emoji} {category}")

            for tool, tests in tools.items():
                sections.append(f"\n#### `{tool}`\n")

                for test in tests:
                    status = "✅" if test.passed else "❌"
                    sections.append(f"**{status} {test.name}** ({test.duration_ms:.0f}ms)")

                    # Parameters
                    params_str = json.dumps(test.input_params, ensure_ascii=False)
                    sections.append(f"\n- **Parámetros:** `{params_str}`")
                    sections.append(f"- **Resultado:** {test.output_summary}")

                    if test.error:
                        sections.append(f"\n<details>\n<summary>❌ Error</summary>\n\n```\n{test.error}\n```\n</details>")

                    if test.output_full and len(test.output_full) < 2000:
                        sections.append(f"\n<details>\n<summary>📄 Respuesta completa</summary>\n\n```json\n{test.output_full[:2000]}\n```\n</details>")

                    sections.append("")

        return "\n".join(sections)

    def _generate_statistics(self) -> str:
        """Generate statistics section."""
        # Latency by tool
        by_tool = {}
        for r in self.results:
            if r.tool not in by_tool:
                by_tool[r.tool] = []
            by_tool[r.tool].append(r.duration_ms)

        stats = """## 📈 Estadísticas de Latencia

| Herramienta | Min (ms) | Max (ms) | Promedio (ms) | Tests |
|-------------|----------|----------|---------------|-------|"""

        for tool, times in sorted(by_tool.items()):
            min_t = min(times)
            max_t = max(times)
            avg_t = sum(times) / len(times)
            stats += f"\n| `{tool}` | {min_t:.0f} | {max_t:.0f} | {avg_t:.0f} | {len(times)} |"

        # Error summary
        errors = [r for r in self.results if not r.passed]
        if errors:
            stats += "\n\n### ❌ Errores Encontrados\n"
            for err in errors:
                stats += f"\n- **{err.tool}** - {err.name}: {err.output_summary}"

        return stats

    def _generate_technical_details(self) -> str:
        """Generate technical details section."""
        import platform

        return f"""## 🔍 Detalles Técnicos

| Detalle | Valor |
|---------|-------|
| Python | {platform.python_version()} |
| Sistema | {platform.system()} {platform.release()} |
| Fecha | {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} |
| Timezone | {datetime.now().astimezone().tzname()} |

---

*Este reporte fue generado automáticamente por `scripts/test_tools_report.py`*"""

    def save(self, path: str):
        """Save report to file."""
        content = self.generate()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        print(f"\n📄 Report saved to: {path}")


async def main():
    """Main entry point."""
    # Run tests
    tester = ToolTester()
    results = await tester.run_all_tests()

    # Generate report
    reporter = MarkdownReporter(results)
    report_path = Path(__file__).parent.parent / "docs" / "test_report.md"
    reporter.save(str(report_path))

    # Exit with appropriate code
    failed = sum(1 for r in results if not r.passed)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
