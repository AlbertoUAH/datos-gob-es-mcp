.PHONY: install dev run test test-cov lint format clean help inspect venv docker-build docker-up docker-down docker-run docker-test docker-dev docker-logs docker-clean notebooks

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

install: venv  ## Install dependencies
	$(VENV_PIP) install -r requirements.txt

dev: venv  ## Install in development mode with dev dependencies
	$(VENV_PIP) install -r requirements-dev.txt

run:  ## Run the MCP server
	$(VENV_PYTHON) server.py

run-stdio:  ## Run the MCP server in stdio mode (for MCP clients)
	$(VENV_DIR)/bin/mcp run server.py

inspect:  ## Inspect the MCP server tools (useful for debugging)
	$(VENV_DIR)/bin/mcp dev server.py

test:  ## Run tests
	$(VENV_PYTHON) -m pytest tests/ -v

test-cov:  ## Run tests with coverage report
	$(VENV_PYTHON) -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

lint:  ## Run linter (ruff)
	$(VENV_PYTHON) -m ruff check server.py prompts/ tests/

format:  ## Format code with ruff
	$(VENV_PYTHON) -m ruff format server.py prompts/ tests/
	$(VENV_PYTHON) -m ruff check --fix server.py prompts/ tests/

clean:  ## Clean up cache and build files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

ci:  ## Run full CI pipeline (lint + test)
	$(MAKE) lint
	$(MAKE) test

# Docker targets
docker-build:  ## Build Docker image with docker-compose
	docker-compose build

docker-up:  ## Start MCP server in Docker (background)
	docker-compose up -d

docker-down:  ## Stop Docker containers
	docker-compose down

docker-run:  ## Run MCP server in Docker (interactive)
	docker run --rm -it --env-file .env datos-gob-es-mcp_mcp-server

docker-test:  ## Test that Docker image works
	docker run --rm --env-file .env datos-gob-es-mcp_mcp-server python -c "import server; print('✓ Server OK')"

docker-dev:  ## Run development server with Docker Compose
	docker-compose --profile dev up mcp-dev

docker-logs:  ## Show Docker container logs
	docker-compose logs -f mcp-server

docker-clean:  ## Remove Docker containers and volumes
	docker-compose down -v --rmi local

# Notebooks
notebooks:  ## Start Jupyter notebook server for examples
	$(VENV_PIP) install jupyter pandas matplotlib
	$(VENV_PYTHON) -m jupyter notebook examples/
