"""MCP server for datos.gob.es open data catalog API."""

__version__ = "0.1.0"

from .client import DatosGobClient, DatosGobClientError
from .server import mcp

__all__ = ["DatosGobClient", "DatosGobClientError", "mcp", "__version__"]
