"""MCP server for BOE (Boletin Oficial del Estado) data."""

from fastmcp import FastMCP
from integrations.boe import register_boe_tools

mcp = FastMCP(
    "boe",
    instructions=(
        "Acceso al BOE (Boletín Oficial del Estado) de España. "
        "Permite buscar y consultar legislación, disposiciones y anuncios oficiales. "
        "Usa boe_search para buscar en el BOE y boe_get para obtener documentos completos."
    ),
)

register_boe_tools(mcp)
