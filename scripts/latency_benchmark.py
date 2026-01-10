#!/usr/bin/env python3
"""Latency benchmark for all MCP tools.

This script measures the latency of all 11 MCP tools in the datos-gob-es server,
running multiple tests per tool to get statistical measures.
"""

import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import tools and clients
from server import get, search, client
from integrations.ine import INEClient
from integrations.aemet import AEMETClient
from integrations.boe import BOEClient


def get_tool_fn(tool):
    """Extract the underlying async function from a FastMCP tool."""
    if hasattr(tool, 'fn'):
        return tool.fn
    return tool


# Get the actual callable functions
search_fn = get_tool_fn(search)
get_fn = get_tool_fn(get)

# Configuration
NUM_RUNS = 5
WARMUP_RUNS = 1
COOLDOWN_SECONDS = 1.0
TIMEOUT_SECONDS = 60


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    tool_name: str
    test_case: str
    runs: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.runs)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0

    @property
    def min_latency(self) -> float | None:
        return min(self.runs) if self.runs else None

    @property
    def max_latency(self) -> float | None:
        return max(self.runs) if self.runs else None

    @property
    def avg_latency(self) -> float | None:
        return statistics.mean(self.runs) if self.runs else None

    @property
    def median_latency(self) -> float | None:
        return statistics.median(self.runs) if self.runs else None

    @property
    def std_dev(self) -> float | None:
        return statistics.stdev(self.runs) if len(self.runs) > 1 else 0

    @property
    def p95_latency(self) -> float | None:
        if len(self.runs) < 2:
            return self.max_latency
        sorted_runs = sorted(self.runs)
        idx = int(len(sorted_runs) * 0.95)
        return sorted_runs[min(idx, len(sorted_runs) - 1)]


class LatencyBenchmark:
    """Benchmark runner for MCP tools."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []
        self.ine_client = INEClient()
        self.aemet_client = AEMETClient()
        self.boe_client = BOEClient()

    async def run_single_test(
        self,
        func: Callable[..., Coroutine],
        *args,
        **kwargs
    ) -> tuple[float | None, str | None]:
        """Run a single test and return (latency_ms, error)."""
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=TIMEOUT_SECONDS
            )
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

            # Check if result indicates an error
            if isinstance(result, str):
                try:
                    data = json.loads(result)
                    if "error" in data:
                        return None, data["error"]
                except json.JSONDecodeError:
                    pass

            return elapsed, None
        except asyncio.TimeoutError:
            return None, f"Timeout after {TIMEOUT_SECONDS}s"
        except Exception as e:
            return None, str(e)

    async def benchmark_tool(
        self,
        tool_name: str,
        test_case: str,
        func: Callable[..., Coroutine],
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Run benchmark for a single tool configuration."""
        result = BenchmarkResult(tool_name=tool_name, test_case=test_case)

        print(f"  Testing {tool_name} ({test_case})...", end=" ", flush=True)

        # Warmup runs
        for _ in range(WARMUP_RUNS):
            await self.run_single_test(func, *args, **kwargs)
            await asyncio.sleep(COOLDOWN_SECONDS)

        # Actual benchmark runs
        for i in range(NUM_RUNS):
            latency, error = await self.run_single_test(func, *args, **kwargs)

            if error:
                result.errors.append(error)
            else:
                result.runs.append(latency)

            # Cooldown between runs
            if i < NUM_RUNS - 1:
                await asyncio.sleep(COOLDOWN_SECONDS)

        if result.avg_latency:
            print(f"avg={result.avg_latency:.0f}ms")
        else:
            print(f"FAILED ({result.errors[0] if result.errors else 'unknown'})")

        self.results.append(result)
        return result

    async def run_datos_gob_benchmarks(self):
        """Run benchmarks for datos.gob.es tools."""
        print("\n=== datos.gob.es Tools ===")

        # search - title
        await self.benchmark_tool(
            "search", "title=empleo",
            search_fn, title="empleo"
        )

        # search - theme
        await self.benchmark_tool(
            "search", "theme=economia",
            search_fn, theme="economia"
        )

        # search - keyword
        await self.benchmark_tool(
            "search", "keyword=presupuesto",
            search_fn, keyword="presupuesto"
        )

        # search - semantic (if embeddings available)
        await self.benchmark_tool(
            "search", "query=desempleo juvenil (semantic)",
            search_fn, query="desempleo juvenil"
        )

        # get - metadata only
        # First, search to get a valid dataset ID
        search_result = await search_fn(title="empleo", page=0)
        dataset_id = None
        try:
            data = json.loads(search_result)
            if data.get("datasets"):
                dataset_id = data["datasets"][0].get("id")
        except:
            pass

        if dataset_id:
            await self.benchmark_tool(
                "get", f"metadata only (id={dataset_id[:20]}...)",
                get_fn, dataset_id=dataset_id
            )

            # get - with data download
            await self.benchmark_tool(
                "get", "include_data=True",
                get_fn, dataset_id=dataset_id, include_data=True, max_rows=100
            )
        else:
            print("  Skipping 'get' tests - no dataset found")

    async def run_ine_benchmarks(self):
        """Run benchmarks for INE tools."""
        print("\n=== INE Tools ===")

        # Create wrapped functions that call the client directly
        async def ine_search_query(query: str):
            operations = await self.ine_client.list_operations()
            query_lower = query.lower()
            filtered = [op for op in operations if query_lower in op.get("Nombre", "").lower()]
            return json.dumps({"operations": filtered[:10]})

        async def ine_search_tables(operation_id: str):
            tables = await self.ine_client.list_tables(operation_id)
            return json.dumps({"tables": tables[:10]})

        async def ine_download_data(table_id: str, n_last: int = 5):
            data = await self.ine_client.get_table_data(table_id, n_last=n_last)
            return json.dumps({"data": data[:10]})

        # ine_search - query
        await self.benchmark_tool(
            "ine_search", "query=empleo",
            ine_search_query, "empleo"
        )

        # ine_search - operation_id (EPA = 30308)
        await self.benchmark_tool(
            "ine_search", "operation_id=30308 (EPA)",
            ine_search_tables, "30308"
        )

        # ine_download - table 4247 (unemployment rate)
        await self.benchmark_tool(
            "ine_download", "table_id=4247 n_last=5",
            ine_download_data, "4247", 5
        )

    async def run_aemet_benchmarks(self):
        """Run benchmarks for AEMET tools."""
        print("\n=== AEMET Tools ===")

        if not os.getenv("AEMET_API_KEY"):
            print("  Skipping AEMET tests - AEMET_API_KEY not set")
            return

        # aemet_list_locations - municipalities
        async def list_municipalities():
            data = await self.aemet_client.get_municipalities()
            return json.dumps({"municipalities": data[:10]})

        await self.benchmark_tool(
            "aemet_list_locations", "municipalities",
            list_municipalities
        )

        # aemet_list_locations - stations
        async def list_stations():
            data = await self.aemet_client.get_stations()
            return json.dumps({"stations": data[:10]})

        await self.benchmark_tool(
            "aemet_list_locations", "stations",
            list_stations
        )

        # aemet_get_forecast - Madrid (28079)
        async def get_forecast():
            data = await self.aemet_client.get_forecast_daily("28079")
            return json.dumps(data)

        await self.benchmark_tool(
            "aemet_get_forecast", "municipality=28079 (Madrid)",
            get_forecast
        )

        # aemet_get_observations - all stations
        async def get_observations():
            data = await self.aemet_client.get_observations()
            return json.dumps(data[:10] if isinstance(data, list) else data)

        await self.benchmark_tool(
            "aemet_get_observations", "all stations",
            get_observations
        )

    async def run_boe_benchmarks(self):
        """Run benchmarks for BOE tools."""
        print("\n=== BOE Tools ===")

        # boe_get_summary - most recent
        async def get_summary():
            from datetime import timedelta
            today = datetime.now()
            for i in range(7):
                check_date = today - timedelta(days=i)
                date_str = check_date.strftime("%Y%m%d")
                try:
                    data = await self.boe_client.get_summary(date_str)
                    if data:
                        return json.dumps({"date": date_str, "sections": len(data.get("data", {}).get("sumario", {}).get("diario", []))})
                except:
                    continue
            return json.dumps({"error": "No BOE found"})

        await self.benchmark_tool(
            "boe_get_summary", "most recent",
            get_summary
        )

        # boe_get_document - sample document
        async def get_document():
            # Try to get a recent document ID from summary
            from datetime import timedelta
            today = datetime.now()
            for i in range(7):
                check_date = today - timedelta(days=i)
                date_str = check_date.strftime("%Y%m%d")
                try:
                    summary = await self.boe_client.get_summary(date_str)
                    if summary:
                        # Extract first document ID
                        sumario = summary.get("data", {}).get("sumario", {})
                        diarios = sumario.get("diario", [])
                        if diarios:
                            for diario in diarios:
                                secciones = diario.get("seccion", [])
                                if secciones:
                                    for seccion in secciones:
                                        departamentos = seccion.get("departamento", [])
                                        if departamentos:
                                            for dept in departamentos:
                                                epigrafes = dept.get("epigrafe", [])
                                                if epigrafes:
                                                    for ep in epigrafes:
                                                        items = ep.get("item", [])
                                                        if items:
                                                            doc_id = items[0].get("id")
                                                            if doc_id:
                                                                data = await self.boe_client.get_document(doc_id)
                                                                return json.dumps({"id": doc_id})
                except:
                    continue
            return json.dumps({"error": "No document found"})

        await self.benchmark_tool(
            "boe_get_document", "recent document",
            get_document
        )

        # boe_search - simple query
        async def boe_search_query():
            from datetime import timedelta
            today = datetime.now()
            date_to = today.strftime("%Y%m%d")
            date_from = (today - timedelta(days=7)).strftime("%Y%m%d")

            # Simple search implementation
            results = []
            for i in range(3):  # Search last 3 days only for benchmark
                check_date = today - timedelta(days=i)
                date_str = check_date.strftime("%Y%m%d")
                try:
                    summary = await self.boe_client.get_summary(date_str)
                    if summary:
                        results.append(date_str)
                except:
                    continue
            return json.dumps({"dates_searched": results})

        await self.benchmark_tool(
            "boe_search", "query=ley (last 3 days)",
            boe_search_query
        )

    async def run_all_benchmarks(self):
        """Run all benchmarks."""
        print(f"\n{'='*60}")
        print(f"MCP Tools Latency Benchmark")
        print(f"{'='*60}")
        print(f"Configuration: {NUM_RUNS} runs, {WARMUP_RUNS} warmup, {COOLDOWN_SECONDS}s cooldown")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        await self.run_datos_gob_benchmarks()
        await self.run_ine_benchmarks()
        await self.run_aemet_benchmarks()
        await self.run_boe_benchmarks()

        print(f"\n{'='*60}")
        print(f"Benchmark completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total tests: {len(self.results)}")

    def generate_report(self) -> str:
        """Generate markdown report from results."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Informe de Latencia - MCP Tools datos-gob-es

**Fecha de ejecucion:** {now}
**Configuracion:** {NUM_RUNS} runs, {WARMUP_RUNS} warmup, {COOLDOWN_SECONDS}s cooldown
**Timeout:** {TIMEOUT_SECONDS}s por operacion

---

## Resumen Ejecutivo

"""
        # Summary table
        report += "| Tool | Test Case | Avg (ms) | Min (ms) | Max (ms) | Std Dev | P95 (ms) | Success | Clasificacion |\n"
        report += "|------|-----------|----------|----------|----------|---------|----------|---------|---------------|\n"

        for r in self.results:
            if r.avg_latency:
                # Classification
                if r.avg_latency < 500:
                    classification = "🟢 Rapido"
                elif r.avg_latency < 2000:
                    classification = "🟡 Moderado"
                else:
                    classification = "🔴 Lento"

                report += f"| {r.tool_name} | {r.test_case} | {r.avg_latency:.0f} | {r.min_latency:.0f} | {r.max_latency:.0f} | {r.std_dev:.0f} | {r.p95_latency:.0f} | {r.success_rate:.0f}% | {classification} |\n"
            else:
                report += f"| {r.tool_name} | {r.test_case} | - | - | - | - | - | {r.success_rate:.0f}% | ❌ Error |\n"

        # Detailed analysis by API
        report += "\n---\n\n## Analisis Detallado por API\n"

        # Group by API
        apis = {
            "datos.gob.es": ["search", "get"],
            "INE": ["ine_search", "ine_download"],
            "AEMET": ["aemet_list_locations", "aemet_get_forecast", "aemet_get_observations"],
            "BOE": ["boe_get_summary", "boe_get_document", "boe_search"],
        }

        for api_name, tools in apis.items():
            api_results = [r for r in self.results if r.tool_name in tools]
            if not api_results:
                continue

            report += f"\n### {api_name}\n\n"

            avg_latencies = [r.avg_latency for r in api_results if r.avg_latency]
            if avg_latencies:
                overall_avg = statistics.mean(avg_latencies)
                report += f"**Latencia promedio general:** {overall_avg:.0f} ms\n\n"

            for r in api_results:
                report += f"#### {r.tool_name} ({r.test_case})\n\n"
                if r.avg_latency:
                    report += f"- **Promedio:** {r.avg_latency:.0f} ms\n"
                    report += f"- **Rango:** {r.min_latency:.0f} - {r.max_latency:.0f} ms\n"
                    report += f"- **Desviacion estandar:** {r.std_dev:.0f} ms\n"
                    report += f"- **Percentil 95:** {r.p95_latency:.0f} ms\n"
                    report += f"- **Runs exitosos:** {r.success_count}/{r.success_count + r.error_count}\n"
                else:
                    report += f"- **Error:** {r.errors[0] if r.errors else 'Unknown'}\n"
                report += "\n"

        # Analysis of causes
        report += "---\n\n## Analisis de Causas de Latencia\n\n"

        slow_tools = [r for r in self.results if r.avg_latency and r.avg_latency >= 2000]
        moderate_tools = [r for r in self.results if r.avg_latency and 500 <= r.avg_latency < 2000]

        report += "### Herramientas Lentas (>2000ms)\n\n"
        if slow_tools:
            for r in slow_tools:
                report += f"**{r.tool_name} ({r.test_case}):** {r.avg_latency:.0f}ms\n\n"

                # Analyze causes based on tool type
                if "search" in r.tool_name and "semantic" in r.test_case.lower():
                    report += "- **Causa probable:** Carga de modelo de embeddings + calculo de similitud semantica\n"
                    report += "- **Detalle:** La busqueda semantica requiere cargar el modelo sentence-transformers y calcular embeddings para todos los candidatos\n"
                elif "download" in r.tool_name or "include_data" in r.test_case:
                    report += "- **Causa probable:** Descarga de datos desde servidor externo\n"
                    report += "- **Detalle:** El tiempo depende del tamano del archivo y velocidad del servidor\n"
                elif "aemet" in r.tool_name:
                    report += "- **Causa probable:** API de AEMET con doble request (metadatos + datos)\n"
                    report += "- **Detalle:** AEMET requiere dos llamadas HTTP por operacion (obtener URL, luego datos)\n"
                elif "boe" in r.tool_name:
                    report += "- **Causa probable:** Procesamiento de XML/JSON del BOE\n"
                    report += "- **Detalle:** Los sumarios del BOE son documentos grandes que requieren parsing\n"
                else:
                    report += "- **Causa probable:** Latencia de red o procesamiento del servidor\n"
                report += "\n"
        else:
            report += "No hay herramientas con latencia >2000ms.\n\n"

        report += "### Herramientas Moderadas (500-2000ms)\n\n"
        if moderate_tools:
            for r in moderate_tools:
                report += f"- **{r.tool_name} ({r.test_case}):** {r.avg_latency:.0f}ms\n"
        else:
            report += "No hay herramientas en este rango.\n"

        # Improvement proposals
        report += "\n---\n\n## Propuestas de Mejora\n\n"
        report += "| Prioridad | Tool | Problema | Propuesta | Impacto Estimado | Esfuerzo |\n"
        report += "|-----------|------|----------|-----------|------------------|----------|\n"

        # Generate proposals based on results
        for r in sorted(self.results, key=lambda x: -(x.avg_latency or 0)):
            if not r.avg_latency:
                continue

            if r.avg_latency >= 2000:
                priority = "🔴 Alta"
                if "semantic" in r.test_case.lower():
                    report += f"| {priority} | {r.tool_name} | Latencia {r.avg_latency:.0f}ms | Cache de embeddings precalculados | -70% latencia | Medio |\n"
                elif "include_data" in r.test_case:
                    report += f"| {priority} | {r.tool_name} | Latencia {r.avg_latency:.0f}ms | Streaming de datos + cache local | -50% latencia | Alto |\n"
                elif "aemet" in r.tool_name:
                    report += f"| {priority} | {r.tool_name} | Latencia {r.avg_latency:.0f}ms | Cache de respuestas AEMET (1h TTL) | -80% latencia | Bajo |\n"
                elif "boe" in r.tool_name:
                    report += f"| {priority} | {r.tool_name} | Latencia {r.avg_latency:.0f}ms | Cache de sumarios BOE (24h TTL) | -90% latencia | Bajo |\n"
                else:
                    report += f"| {priority} | {r.tool_name} | Latencia {r.avg_latency:.0f}ms | Analizar bottleneck especifico | Variable | Medio |\n"
            elif r.avg_latency >= 1000:
                priority = "🟡 Media"
                if "ine" in r.tool_name:
                    report += f"| {priority} | {r.tool_name} | Latencia {r.avg_latency:.0f}ms | Cache de operaciones INE (24h TTL) | -60% latencia | Bajo |\n"
                elif "list" in r.tool_name:
                    report += f"| {priority} | {r.tool_name} | Latencia {r.avg_latency:.0f}ms | Cache de listados estaticos | -80% latencia | Bajo |\n"
                else:
                    report += f"| {priority} | {r.tool_name} | Latencia {r.avg_latency:.0f}ms | Optimizar parsing de respuesta | -20% latencia | Bajo |\n"

        # Technical recommendations
        report += "\n---\n\n## Recomendaciones Tecnicas\n\n"
        report += """### 1. Implementar Cache Multi-nivel

```
Nivel 1: Cache en memoria (LRU, 5 min TTL) - Respuestas frecuentes
Nivel 2: Cache en disco (24h TTL) - Metadatos estaticos
Nivel 3: Redis/Memcached - Para despliegues multi-instancia
```

### 2. Optimizaciones de Red

- **Connection pooling:** Ya implementado con httpx
- **HTTP/2:** Ya habilitado para multiplexing
- **Keep-alive:** Configurado para reutilizar conexiones
- **Compresion:** Usar Accept-Encoding: gzip

### 3. Optimizaciones de Embeddings (Busqueda Semantica)

- **Pre-cargar modelo:** En startup del servidor (PRELOAD_EMBEDDINGS_MODEL=true)
- **Cache de embeddings:** Guardar vectores calculados en disco
- **Reducir candidatos:** Limitar MAX_SEMANTIC_CANDIDATES

### 4. Paralelizacion

- **Busquedas multi-tema:** Ya implementado con asyncio.gather
- **Paginacion paralela:** Ya implementado (PARALLEL_PAGES=5)

---

## Conclusiones

"""
        # Calculate overall stats
        all_latencies = [r.avg_latency for r in self.results if r.avg_latency]
        if all_latencies:
            report += f"- **Latencia promedio global:** {statistics.mean(all_latencies):.0f} ms\n"
            report += f"- **Latencia mediana global:** {statistics.median(all_latencies):.0f} ms\n"
            report += f"- **Herramienta mas rapida:** {min(self.results, key=lambda x: x.avg_latency or float('inf')).tool_name} ({min(all_latencies):.0f} ms)\n"
            report += f"- **Herramienta mas lenta:** {max(self.results, key=lambda x: x.avg_latency or 0).tool_name} ({max(all_latencies):.0f} ms)\n"

            fast = len([l for l in all_latencies if l < 500])
            moderate = len([l for l in all_latencies if 500 <= l < 2000])
            slow = len([l for l in all_latencies if l >= 2000])

            report += f"\n**Distribucion de rendimiento:**\n"
            report += f"- 🟢 Rapido (<500ms): {fast} tools ({fast/len(all_latencies)*100:.0f}%)\n"
            report += f"- 🟡 Moderado (500-2000ms): {moderate} tools ({moderate/len(all_latencies)*100:.0f}%)\n"
            report += f"- 🔴 Lento (>2000ms): {slow} tools ({slow/len(all_latencies)*100:.0f}%)\n"

        return report


async def main():
    """Main entry point."""
    benchmark = LatencyBenchmark()
    await benchmark.run_all_benchmarks()

    # Generate and save report
    report = benchmark.generate_report()

    report_path = Path(__file__).parent.parent / "docs" / "latency_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"\n📊 Report saved to: {report_path}")

    # Also print summary to stdout
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for r in benchmark.results:
        if r.avg_latency:
            status = "🟢" if r.avg_latency < 500 else ("🟡" if r.avg_latency < 2000 else "🔴")
            print(f"{status} {r.tool_name:25} {r.test_case:30} {r.avg_latency:>8.0f}ms")
        else:
            print(f"❌ {r.tool_name:25} {r.test_case:30} {'ERROR':>8}")


if __name__ == "__main__":
    asyncio.run(main())
