"""Base connection interface for the MCP Aggregator."""

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import logging

from aggregator.config import ServerSpec

logger = logging.getLogger(__name__)


@dataclass
class Downstream:
    """Container for a connected downstream MCP server."""

    spec: ServerSpec
    session: Any  # mcp.ClientSession


class MCPConnection(ABC):
    """Abstract base class for MCP connections."""

    def __init__(self, spec: ServerSpec, repo_root: Path):
        if not repo_root.is_absolute():
            raise ValueError(f"repo_root must be absolute path, got {repo_root}")
        if not repo_root.exists():
            raise ValueError(f"repo_root does not exist: {repo_root}")
        self.spec = spec
        self.repo_root = repo_root
        self.mcp_path: Optional[Path] = None

    @abstractmethod
    async def validate(self) -> bool:
        pass

    @abstractmethod
    async def connect(self, stack: AsyncExitStack) -> Downstream:
        pass
