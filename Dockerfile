# datos-gob-es-mcp Docker image
# MCP Server for Spanish Open Data (datos.gob.es)

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------
# Production image
# -----------------------------------------------------------
FROM base AS production

# Copy application code
COPY server.py .
COPY core/ core/
COPY prompts/ prompts/
COPY integrations/ integrations/
COPY notifications/ notifications/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /home/appuser/.cache/datos-gob-es && \
    chown -R appuser:appuser /app /home/appuser/.cache

USER appuser

# Set cache directory
ENV HOME=/home/appuser

# Health check endpoint (optional - for orchestrators)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import server; print('OK')" || exit 1

# Default command - run MCP server in stdio mode
CMD ["fastmcp", "run", "server.py"]

# -----------------------------------------------------------
# Development image with additional tools
# -----------------------------------------------------------
FROM base AS development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy all application code
COPY . .

# Development command
CMD ["fastmcp", "dev", "server.py"]
