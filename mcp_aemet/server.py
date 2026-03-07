"""MCP server for AEMET (Agencia Estatal de Meteorologia) data."""

from fastmcp import FastMCP
from integrations.aemet import register_aemet_tools

mcp = FastMCP(
    "aemet",
    instructions=(
        "Acceso a datos meteorológicos de AEMET (Agencia Estatal de Meteorología de España). "
        "Proporciona predicciones, observaciones y datos climáticos. "
        "Requiere la variable de entorno AEMET_API_KEY para funcionar."
    ),
)

register_aemet_tools(mcp)
