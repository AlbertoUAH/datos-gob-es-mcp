"""MCP server for INE (Instituto Nacional de Estadistica) data."""

from fastmcp import FastMCP
from integrations.ine import register_ine_tools

mcp = FastMCP(
    "ine",
    instructions=(
        "Acceso a datos estadísticos del INE (Instituto Nacional de Estadística de España). "
        "Permite buscar y descargar tablas estadísticas del sistema PC-Axis del INE. "
        "Usa ine_search para buscar tablas y ine_download para obtener los datos."
    ),
)

register_ine_tools(mcp)
