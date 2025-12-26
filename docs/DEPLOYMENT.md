# Deployment Guide

This guide covers different deployment options for the datos-gob-es-mcp server.

## Quick Start with Docker

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+ (optional)

### Build and Run

```bash
# Build the image
docker build -t datos-gob-es-mcp .

# Run the container
docker run -it --rm datos-gob-es-mcp

# With AEMET API key
docker run -it --rm -e AEMET_API_KEY=your_key_here datos-gob-es-mcp
```

### Using Docker Compose

```bash
# Create .env file with your API keys
cp .env.example .env
# Edit .env and add your AEMET_API_KEY

# Start the server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AEMET_API_KEY` | For weather | AEMET OpenData API key ([get free key](https://opendata.aemet.es/centrodedescargas/altaUsuario)) |
| `WEBHOOK_SECRET` | No | Secret for webhook signature validation |

## Deployment Options

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
mcp run server.py

# Or with the Makefile
make run-stdio
```

### 2. Docker (Recommended for Production)

The Docker image is optimized for production:
- Multi-stage build for smaller image size
- Non-root user for security
- Health check included
- Persistent cache volume

```bash
# Production build
docker build --target production -t datos-gob-es-mcp:prod .

# Run with persistent cache
docker run -it --rm \
  -v datos-gob-es-cache:/home/appuser/.cache/datos-gob-es \
  -e AEMET_API_KEY=your_key \
  datos-gob-es-mcp:prod
```

### 3. FastMCP Cloud

The server is ready for FastMCP Cloud deployment:

```bash
# Deploy to FastMCP Cloud
fastmcp deploy server.py
```

### 4. Kubernetes

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: datos-gob-es-mcp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: datos-gob-es-mcp
  template:
    metadata:
      labels:
        app: datos-gob-es-mcp
    spec:
      containers:
      - name: mcp-server
        image: datos-gob-es-mcp:latest
        env:
        - name: AEMET_API_KEY
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: aemet-api-key
        resources:
          limits:
            cpu: "1"
            memory: "512Mi"
          requests:
            cpu: "250m"
            memory: "128Mi"
        volumeMounts:
        - name: cache
          mountPath: /home/appuser/.cache/datos-gob-es
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: mcp-cache-pvc
```

## Client Configuration

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "datos-gob-es": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "datos-gob-es-mcp"]
    }
  }
}
```

Or with local installation:

```json
{
  "mcpServers": {
    "datos-gob-es": {
      "command": "mcp",
      "args": ["run", "/path/to/datos-gob-es-mcp/server.py"]
    }
  }
}
```

### Cursor / VS Code

Add to MCP settings:

```json
{
  "mcp.servers": {
    "datos-gob-es": {
      "command": "mcp",
      "args": ["run", "/path/to/server.py"]
    }
  }
}
```

## Monitoring

### Health Check

The Docker image includes a health check that verifies the server can import correctly:

```bash
docker inspect --format='{{.State.Health.Status}}' datos-gob-es-mcp
```

### Logs

```bash
# Docker logs
docker logs datos-gob-es-mcp

# Docker Compose logs
docker-compose logs -f mcp-server
```

## Troubleshooting

### Common Issues

1. **AEMET API returns 401 Unauthorized**
   - Verify your API key is correct
   - Keys expire; get a new one if needed

2. **Webhook not receiving notifications**
   - Check that your endpoint is publicly accessible
   - Verify firewall rules allow incoming POST requests
   - Check webhook_list to see if it's registered

3. **Slow first semantic search**
   - First search builds the embedding index (~30-60 seconds)
   - Subsequent searches are fast (<1 second)
   - Index is cached to disk

4. **Docker container exits immediately**
   - Use `-it` flags for interactive mode
   - Check logs: `docker logs <container_id>`

### Getting Help

- [GitHub Issues](https://github.com/AlbertoUAH/datos-gob-es-mcp/issues)
- [MCP Documentation](https://modelcontextprotocol.io/)
