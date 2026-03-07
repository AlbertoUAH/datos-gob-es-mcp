"""SSE-based MCP connection for the aggregator."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession
from mcp.client.sse import sse_client

from aggregator.connections.base import Downstream, MCPConnection
from aggregator.config import ServerSpec, Transport

logger = logging.getLogger(__name__)


class SSEMCPConnection(MCPConnection):
    """Connects to a remote MCP server via SSE/HTTP."""

    def __init__(self, spec: ServerSpec, repo_root: Path):
        super().__init__(spec, repo_root)
        if spec.transport not in ("streamable-http", Transport.STREAMABLE_HTTP):
            raise ValueError(
                f"SSEMCPConnection requires transport='streamable-http', got '{spec.transport}'"
            )
        if not spec.url:
            raise ValueError(f"SSEMCPConnection requires 'url' field for '{spec.name}'")
        self.url = spec.url

    async def validate(self) -> bool:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(
                f"[{self.spec.name}] Invalid URL: {self.url}. Must start with http:// or https://"
            )
        return True

    async def connect(self, stack: AsyncExitStack) -> Downstream:
        logger.info(f"[{self.spec.name}] Connecting to SSE endpoint {self.url}...")

        async def _connect():
            read, write = await stack.enter_async_context(sse_client(self.url))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return Downstream(spec=self.spec, session=session)

        try:
            result = await asyncio.wait_for(_connect(), timeout=self.spec.timeout_seconds)
            logger.info(f"[{self.spec.name}] Connection established")
            return result
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Connection to '{self.spec.name}' timed out after {self.spec.timeout_seconds}s"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to '{self.spec.name}': {e}") from e
