"""MCP Connection Factory for the datos.gob.es aggregator."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from pathlib import Path

from aggregator.config import ServerSpec, Transport
from aggregator.connections.base import Downstream, MCPConnection
from aggregator.connections.stdio import STDIOMCPConnection
from aggregator.connections.sse import SSEMCPConnection

logger = logging.getLogger(__name__)


class MCPConnectionFactory:
    """Factory for creating and connecting to MCP servers."""

    def __init__(self, repo_root: Path):
        if not repo_root.is_absolute():
            raise ValueError(f"repo_root must be absolute path, got: {repo_root}")
        if not repo_root.exists():
            raise ValueError(f"repo_root does not exist: {repo_root}")
        self.repo_root = repo_root
        self._stats = {"attempted": 0, "validated": 0, "connected": 0, "failed": 0}
        logger.info(f"MCPConnectionFactory initialized for: {repo_root}")

    def create_connection(self, spec: ServerSpec) -> MCPConnection:
        if spec.transport in ("stdio", Transport.STDIO):
            return STDIOMCPConnection(spec, self.repo_root)
        elif spec.transport in ("streamable-http", Transport.STREAMABLE_HTTP):
            return SSEMCPConnection(spec, self.repo_root)
        raise ValueError(f"Unsupported transport type: {spec.transport}")

    async def create_and_connect(
        self, spec: ServerSpec, stack: AsyncExitStack
    ) -> Downstream:
        self._stats["attempted"] += 1
        logger.info(f"[{spec.name}] Connecting...")

        try:
            connection = self.create_connection(spec)

            await connection.validate()
            self._stats["validated"] += 1

            downstream = await connection.connect(stack)
            self._stats["connected"] += 1
            logger.info(f"[{spec.name}] Connected successfully")
            return downstream

        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"[{spec.name}] Connection failed: {e}")
            raise

    def get_stats(self) -> dict:
        return self._stats.copy()
