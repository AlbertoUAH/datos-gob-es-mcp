"""Entry point for the datos.gob.es MCP Aggregator.

Usage:
    python -m aggregator [--port 8001] [--host 0.0.0.0]
"""

import sys
from pathlib import Path

from aggregator.config import AggregatorConfig
from aggregator.server import run_server


def main():
    project_root = Path(__file__).parents[1].resolve()

    # Parse CLI args
    host = "0.0.0.0"
    port = 8001
    args = sys.argv[1:]
    if "--port" in args:
        port = int(args[args.index("--port") + 1])
    if "--host" in args:
        host = args[args.index("--host") + 1]

    cfg = AggregatorConfig.from_file(
        Path(__file__).parent / "servers.json",
        project_root=str(project_root),
    )

    run_server(cfg, project_root=project_root, host=host, port=port)


if __name__ == "__main__":
    main()
