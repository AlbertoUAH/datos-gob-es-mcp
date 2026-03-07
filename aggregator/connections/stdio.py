"""STDIO-based MCP connection for the aggregator."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from aggregator.connections.base import Downstream, MCPConnection
from aggregator.config import ServerSpec, Transport

logger = logging.getLogger(__name__)


class STDIOMCPConnection(MCPConnection):
    """Launches a local MCP as a subprocess and communicates via stdio."""

    def __init__(self, spec: ServerSpec, repo_root: Path):
        super().__init__(spec, repo_root)
        if spec.transport not in ("stdio", Transport.STDIO):
            raise ValueError(
                f"STDIOMCPConnection requires transport='stdio', got '{spec.transport}'"
            )

    async def validate(self) -> bool:
        """Check that the module directory exists under repo_root."""
        # Extract module name from args (look for -m flag)
        module_name = _extract_module_name(self.spec.args)

        # Search for the MCP at repo root level (our convention) or src/ level (warehouse)
        candidates = [
            self.repo_root / module_name,
            self.repo_root / f"mcp_{module_name}",
            self.repo_root / "src" / module_name,
            self.repo_root / "src" / f"mcp_{module_name}",
        ]
        for path in candidates:
            if path.exists() and (path / "__main__.py").exists():
                self.mcp_path = path
                logger.info(f"[{self.spec.name}] Found MCP at {path}")
                return True

        raise ValueError(
            f"MCP '{self.spec.name}' not found.\n"
            f"Searched paths:\n" + "\n".join(f"  - {p}" for p in candidates)
        )

    async def connect(self, stack: AsyncExitStack) -> Downstream:
        if not self.mcp_path:
            raise RuntimeError(f"[{self.spec.name}] Must call validate() before connect()")

        logger.info(f"[{self.spec.name}] Launching subprocess...")

        async def _connect():
            merged_env = os.environ.copy()
            if self.spec.env:
                merged_env.update(self.spec.env)

            params = StdioServerParameters(
                command=self.spec.command,
                args=self.spec.args,
                env=merged_env,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
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
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Command '{self.spec.command}' not found for '{self.spec.name}'"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to '{self.spec.name}': {e}"
            ) from e


def _extract_module_name(args: list[str]) -> str:
    """Extract module name from args list (handles -m flag pattern)."""
    try:
        idx = args.index("-m")
        if idx + 1 < len(args):
            return args[idx + 1]
    except ValueError:
        pass
    if len(args) == 1:
        return args[0]
    raise ValueError(f"Cannot extract module name from args: {args}")
