.PHONY: install dev run test lint format clean help inspect venv

# Default Python interpreter
PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip

install: venv  ## Install the package
	$(VENV_PIP) install -r requirements.txt 2>/dev/null || $(VENV_PIP) install mcp[cli] httpx pydantic

dev: venv  ## Install in development mode with dev dependencies
	$(VENV_PIP) install mcp[cli] httpx pydantic ruff pytest pytest-asyncio

run:  ## Run the MCP server
	$(VENV_PYTHON) server.py

run-stdio:  ## Run the MCP server in stdio mode (for MCP clients)
	$(VENV_DIR)/bin/mcp run server.py

inspect:  ## Inspect the MCP server tools (useful for debugging)
	$(VENV_DIR)/bin/mcp dev server.py

test:  ## Run tests
	$(VENV_PYTHON) -m pytest tests/ -v

lint:  ## Run linter (ruff)
	$(VENV_PYTHON) -m ruff check server.py

format:  ## Format code with ruff
	$(VENV_PYTHON) -m ruff format server.py
	$(VENV_PYTHON) -m ruff check --fix server.py

clean:  ## Clean up cache and build files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
