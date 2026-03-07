"""Configuration models for the datos.gob.es MCP Aggregator."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Transport(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"


class ServerSpec(BaseModel):
    name: str = Field(..., description="Logical name of the downstream server")
    prefix: Optional[str] = Field(default=None, description="Tool name prefix; defaults to name")
    enabled: bool = True
    transport: Transport = Transport.STDIO

    # STDIO spec
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)

    # HTTP spec
    url: Optional[str] = None

    # Timeout
    timeout_seconds: float = Field(default=240.0, gt=0)

    @model_validator(mode="after")
    def validate_transport_requirements(self) -> "ServerSpec":
        if self.transport == Transport.STDIO and not self.command:
            raise ValueError(
                f"Server '{self.name}': transport='stdio' requires 'command' field."
            )
        elif self.transport == Transport.STREAMABLE_HTTP and not self.url:
            raise ValueError(
                f"Server '{self.name}': transport='streamable-http' requires 'url' field."
            )
        return self


class AggregatorConfig(BaseModel):
    servers: Dict[str, ServerSpec] = Field(default_factory=dict)

    @classmethod
    def from_servers_array(
        cls, arr: List[dict], project_root: str | None = None
    ) -> "AggregatorConfig":
        """Create configuration from a JSON array, resolving ${PROJECT_ROOT} placeholders."""
        servers: Dict[str, ServerSpec] = {}
        for raw in arr:
            if "name" not in raw:
                continue
            if not raw.get("enabled", True):
                continue

            # Resolve ${PROJECT_ROOT} placeholders
            if project_root:
                raw = _resolve_placeholders(raw, {"PROJECT_ROOT": project_root})

            spec = ServerSpec(**raw)
            if spec.prefix is None:
                spec.prefix = spec.name
            servers[spec.name] = spec
        return cls(servers=servers)

    @classmethod
    def from_file(
        cls, path: Path, project_root: str | None = None
    ) -> "AggregatorConfig":
        """Load configuration from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return cls.from_servers_array(data, project_root=project_root)
        raise ValueError(f"Expected JSON array in {path}, got {type(data)}")


def _resolve_placeholders(obj: dict, vars: dict[str, str]) -> dict:
    """Recursively resolve ${VAR} placeholders in a dict, also from environment."""
    import copy

    result = copy.deepcopy(obj)
    _resolve_recursive(result, vars)
    return result


def _resolve_recursive(obj, vars: dict[str, str]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                obj[k] = _resolve_str(v, vars)
            elif isinstance(v, (dict, list)):
                _resolve_recursive(v, vars)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                obj[i] = _resolve_str(v, vars)
            elif isinstance(v, (dict, list)):
                _resolve_recursive(v, vars)


def _resolve_str(s: str, vars: dict[str, str]) -> str:
    import re

    def replace(m):
        key = m.group(1)
        # First check provided vars, then environment
        return vars.get(key) or os.environ.get(key) or m.group(0)

    return re.sub(r"\$\{([^}]+)\}", replace, s)
