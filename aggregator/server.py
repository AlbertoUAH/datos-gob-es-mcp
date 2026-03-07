"""Core aggregation logic for the datos.gob.es MCP Aggregator.

Builds a FastMCP server that discovers tools from all downstream servers and
exposes them with prefixed names. Also provides a unified_search tool that
queries all sources simultaneously.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp import types

from aggregator.connections.base import Downstream
from aggregator.config import AggregatorConfig
from aggregator.factory import MCPConnectionFactory

logger = logging.getLogger(__name__)


@dataclass
class AggregatorState:
    cfg: AggregatorConfig
    stack: AsyncExitStack
    downstreams: list[Downstream] = field(default_factory=list)


def _proxy_tool_name(prefix: Optional[str], tool_name: str) -> str:
    safe = tool_name.replace("-", "_")
    if not prefix:
        return safe
    return f"{prefix}_{safe}"


def _pick_return_payload(result: Any) -> Any:
    try:
        if getattr(result, "structuredContent", None) is not None:
            return result.structuredContent
        texts = []
        for c in getattr(result, "content", []) or []:
            if isinstance(c, types.TextContent) or getattr(c, "type", None) == "text":
                texts.append(getattr(c, "text", ""))
        if texts:
            return "\n".join(t for t in texts if t)
    except Exception:
        pass
    return result


def _to_fastmcp_tool_result(result: Any) -> Any:
    if not isinstance(result, types.CallToolResult):
        return result
    content_blocks = list(getattr(result, "content", []) or [])
    structured_content = getattr(result, "structuredContent", None)
    meta = getattr(result, "meta", None)
    if not content_blocks and structured_content is None:
        return ""
    if content_blocks:
        return ToolResult(content=content_blocks, structured_content=structured_content, meta=meta)
    return ToolResult(content="", structured_content=structured_content, meta=meta)


def _resource_contents_to_payload(read_result: types.ReadResourceResult) -> Any:
    contents = list(getattr(read_result, "contents", []) or [])
    if not contents:
        return ""
    first = contents[0]
    if isinstance(first, types.TextResourceContents) or hasattr(first, "text"):
        return getattr(first, "text", "")
    if isinstance(first, types.BlobResourceContents) or hasattr(first, "blob"):
        return getattr(first, "blob", "")
    return str(first)


def _create_proxy_for_tool(ds: Downstream, t: types.Tool, proxy_name: str):
    """Dynamically generate a proxy function for a downstream tool."""
    input_schema = t.inputSchema or {"type": "object", "properties": {}}
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    def _extract_schema_type(prop_schema: dict) -> Optional[str]:
        schema_type = prop_schema.get("type")
        if isinstance(schema_type, list):
            non_null = [x for x in schema_type if x != "null"]
            return non_null[0] if non_null else None
        elif isinstance(schema_type, str):
            return schema_type
        for combinator in ("anyOf", "oneOf", "allOf"):
            options = prop_schema.get(combinator)
            if not isinstance(options, list):
                continue
            types_ = [_extract_schema_type(o) for o in options if isinstance(o, dict)]
            types_ = [x for x in types_ if x]
            for preferred in ("object", "array", "integer", "number", "boolean", "string"):
                if preferred in types_:
                    return preferred
            if types_:
                return types_[0]
        if "properties" in prop_schema or "additionalProperties" in prop_schema:
            return "object"
        return None

    def get_python_type(prop_schema: dict) -> str:
        schema_type = _extract_schema_type(prop_schema) or "string"
        return {"string": "str", "integer": "int", "number": "float",
                "boolean": "bool", "array": "list", "object": "dict"}.get(schema_type, "str")

    param_defs = []
    for param_name in properties:
        safe = param_name.replace("-", "_").replace(".", "_")
        python_type = get_python_type(properties[param_name])
        if param_name in required:
            param_defs.append(f"{safe}: {python_type}")
        else:
            param_defs.append(f"{safe}: {python_type} = None")

    params_str = ", ".join(param_defs)
    param_names = [p.replace("-", "_").replace(".", "_") for p in properties]

    func_code = f"async def {proxy_name}({params_str}):\n"
    func_code += "    arguments = {}\n"
    for orig, safe in zip(properties, param_names):
        func_code += f"    if {safe} is not None: arguments['{orig}'] = {safe}\n"
    func_code += f"""
    try:
        result = await _ds_session.call_tool('{t.name}', arguments=arguments)
        if getattr(result, 'isError', False):
            import logging; logging.getLogger(__name__).error(f"[{proxy_name}] Tool error: {{result}}")
        return _to_fastmcp_tool_result(result)
    except Exception as e:
        import logging; logging.getLogger(__name__).error(f"[{proxy_name}] Error: {{e}}")
        raise
"""
    namespace = {"_ds_session": ds.session, "_to_fastmcp_tool_result": _to_fastmcp_tool_result}
    exec(func_code, namespace)
    return namespace[proxy_name]


def _register_proxy_tool(mcp: FastMCP, ds: Downstream, t: types.Tool, proxy_name: str) -> None:
    description = t.description or f"Proxy to {ds.spec.name}.{t.name}"
    logger.info(f"Registering proxy tool: {proxy_name} -> {ds.spec.name}.{t.name}")
    proxy_func = _create_proxy_for_tool(ds, t, proxy_name)
    try:
        mcp.tool(
            name=proxy_name,
            title=getattr(t, "title", None),
            description=description,
            annotations=getattr(t, "annotations", None),
        )(proxy_func)
    except Exception as e:
        logger.error(f"Failed to register {proxy_name}: {e}")
        raise


def _register_proxy_resource(mcp: FastMCP, ds: Downstream, r: types.Resource) -> None:
    resource_uri = str(r.uri)

    async def _proxy_resource_handler():
        result = await ds.session.read_resource(resource_uri)
        return _resource_contents_to_payload(result)

    mcp.resource(
        uri=resource_uri,
        name=r.name,
        description=r.description,
        mime_type=r.mimeType,
    )(_proxy_resource_handler)
    logger.info(f"Registered resource proxy: {resource_uri}")


def build_server(cfg: AggregatorConfig, project_root: Path) -> FastMCP:
    """Build a FastMCP aggregator server from the given configuration."""

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        print("=" * 70)
        print("[AGGREGATOR] Starting datos.gob.es MCP Aggregator...")
        print(f"[AGGREGATOR] Project root: {project_root}")
        print("=" * 70)

        stack = AsyncExitStack()
        async with stack:
            factory = MCPConnectionFactory(project_root)
            downstreams: list[Downstream] = []

            for spec in cfg.servers.values():
                try:
                    downstream = await factory.create_and_connect(spec, stack)
                    downstreams.append(downstream)
                    logger.info(f"Connected: {spec.name}")
                except Exception as e:
                    logger.error(f"Failed to connect to '{spec.name}': {e}")
                    raise RuntimeError(
                        f"Failed to connect to server '{spec.name}': {e}"
                    ) from e

            stats = factory.get_stats()
            logger.info(f"Connected {stats['connected']}/{stats['attempted']} servers")

            state = AggregatorState(cfg=cfg, stack=stack, downstreams=downstreams)

            # Discover and register proxy tools + resources
            seen_tools: set[str] = set()
            seen_resources: set[str] = set()
            total_tools = 0
            total_resources = 0

            for ds in downstreams:
                tools_resp = await ds.session.list_tools()
                logger.info(f"  {ds.spec.name}: {len(tools_resp.tools)} tools")
                for tool in tools_resp.tools:
                    proxy_name = _proxy_tool_name(ds.spec.prefix, tool.name)
                    if proxy_name in seen_tools:
                        raise RuntimeError(
                            f"Tool name collision: '{proxy_name}' (server: {ds.spec.name}). "
                            f"Check prefix configuration."
                        )
                    _register_proxy_tool(server, ds, tool, proxy_name=proxy_name)
                    seen_tools.add(proxy_name)
                    total_tools += 1

                try:
                    resources_resp = await ds.session.list_resources()
                    for resource in resources_resp.resources:
                        resource_uri = str(resource.uri)
                        if resource_uri not in seen_resources:
                            _register_proxy_resource(server, ds, resource)
                            seen_resources.add(resource_uri)
                            total_resources += 1
                except Exception as exc:
                    logger.debug(f"Skipping resources for {ds.spec.name}: {exc}")

            # Register unified_search after all proxy tools
            _register_unified_search(server, state)

            logger.info(f"Registered {total_tools} proxy tools, {total_resources} resources")
            print(f"[AGGREGATOR] Ready: {total_tools} tools, {total_resources} resources")
            print("[AGGREGATOR] unified_search tool available")

            try:
                yield state
            finally:
                logger.info("Aggregator shutting down...")

    return FastMCP("datos-gob-es-aggregator", lifespan=lifespan)


def _register_unified_search(mcp: FastMCP, state: AggregatorState) -> None:
    """Register the unified_search meta-tool on the aggregator server."""

    # Map from source name to tool name in that source
    SEARCH_TOOL_MAP = {
        "datos_gob": "search",
        "ine": "ine_search",
        "boe": "boe_search",
        "eurostat": "eurostat_search",
    }

    async def unified_search(
        query: str,
        sources: list = None,
    ) -> str:
        """Search simultaneously across datos.gob.es, INE, BOE, and Eurostat.

        Args:
            query: Search query (supports Spanish and English).
            sources: Optional list of sources to include.
                     Options: 'datos_gob', 'ine', 'boe', 'eurostat'.
                     Default: all available sources.

        Returns:
            JSON with combined results labeled by source.
        """
        active_sources = sources or list(SEARCH_TOOL_MAP.keys())

        # Build tasks for each active downstream
        tasks: dict[str, asyncio.Task] = {}
        for ds in state.downstreams:
            if ds.spec.name not in active_sources:
                continue
            tool_name = SEARCH_TOOL_MAP.get(ds.spec.name)
            if not tool_name:
                continue
            coro = ds.session.call_tool(tool_name, {"query": query})
            tasks[ds.spec.name] = asyncio.ensure_future(coro)

        if not tasks:
            return json.dumps(
                {"error": "No active sources matched", "available": list(SEARCH_TOOL_MAP.keys())},
                ensure_ascii=False,
            )

        results_raw = await asyncio.gather(*tasks.values(), return_exceptions=True)

        combined: dict[str, Any] = {"query": query, "sources": {}}
        for source_name, result in zip(tasks.keys(), results_raw):
            if isinstance(result, Exception):
                combined["sources"][source_name] = {"error": str(result)}
                continue
            payload = _pick_return_payload(result)
            try:
                if isinstance(payload, str):
                    payload = json.loads(payload)
            except Exception:
                pass
            combined["sources"][source_name] = payload

        return json.dumps(combined, ensure_ascii=False, indent=2)

    mcp.tool(
        name="unified_search",
        description=(
            "Search simultaneously across datos.gob.es, INE, BOE, and Eurostat. "
            "Returns combined results labeled by source."
        ),
    )(unified_search)
    logger.info("Registered unified_search tool")


def run_server(
    cfg: AggregatorConfig,
    project_root: Path,
    host: str = "0.0.0.0",
    port: int = 8001,
) -> None:
    """Start the MCP Aggregator with SSE transport.

    Clients connect to: http://<host>:<port>/sse
    """
    mcp = build_server(cfg, project_root)
    print("=" * 60)
    print(f"[AGGREGATOR] SSE endpoint: http://{host}:{port}/sse")
    print("=" * 60)
    mcp.run(transport="sse", host=host, port=port)
