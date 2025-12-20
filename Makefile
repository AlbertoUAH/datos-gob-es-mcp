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
	$(VENV_PIP) install .

dev: venv  ## Install in development mode with dev dependencies
	$(VENV_PIP) install -e ".[dev]"

run:  ## Run the MCP server
	$(VENV_PYTHON) -m datos_gob_es_mcp.server

run-stdio:  ## Run the MCP server in stdio mode (for MCP clients)
	$(VENV_DIR)/bin/mcp run src/datos_gob_es_mcp/server.py

inspect:  ## Inspect the MCP server tools (useful for debugging)
	$(VENV_DIR)/bin/mcp dev src/datos_gob_es_mcp/server.py

test:  ## Run tests
	$(VENV_PYTHON) -m pytest tests/ -v

lint:  ## Run linter (ruff)
	$(VENV_PYTHON) -m ruff check src/

format:  ## Format code with ruff
	$(VENV_PYTHON) -m ruff format src/
	$(VENV_PYTHON) -m ruff check --fix src/

clean:  ## Clean up cache and build files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

build:  ## Build the package
	$(VENV_PIP) install build
	$(VENV_PYTHON) -m build

publish-test:  ## Publish to TestPyPI
	$(VENV_PIP) install twine
	$(VENV_PYTHON) -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	$(VENV_PIP) install twine
	$(VENV_PYTHON) -m twine upload dist/*
