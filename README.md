# datos-gob-es-mcp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

Servidor MCP (Model Context Protocol) para acceder al catálogo de datos abiertos de España a través de la API de **datos.gob.es**.

## Descripción

Este servidor MCP permite a asistentes de IA como Claude, ChatGPT y otros clientes compatibles con MCP buscar y explorar los miles de datasets públicos disponibles en el portal de datos abiertos del Gobierno de España.

### Características

- 22 herramientas MCP para consultar la API de datos.gob.es
- Búsqueda de datasets por título, temática, publicador, formato, keywords y más
- Acceso a distribuciones (archivos descargables) de los datasets
- Consulta de metadatos: publicadores, temáticas, cobertura geográfica
- Información territorial según la Norma Técnica de Interoperabilidad (NTI)
- Cliente HTTP asíncrono con manejo robusto de errores
- Modelos Pydantic para tipado seguro
- Listo para desplegar en FastMCP Cloud

## Instalación

### Requisitos

- Python 3.10 o superior
- pip

### Instalación rápida

```bash
# Clonar el repositorio
git clone https://github.com/AlbertoUAH/datos-gob-es-mcp.git
cd datos-gob-es-mcp

# Crear entorno virtual e instalar
make dev
```

### Instalación manual

```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar en modo desarrollo
pip install -e ".[dev]"
```

## Uso

### Ejecutar el servidor MCP

```bash
# Modo stdio (para clientes MCP)
make run-stdio

# O directamente
mcp run src/datos_gob_es_mcp/server.py
```

### Inspeccionar herramientas disponibles

```bash
make inspect
```

## Herramientas Disponibles

### Datasets (9 herramientas)

| Herramienta | Descripción |
|-------------|-------------|
| `list_datasets` | Listar datasets con paginación y ordenación |
| `get_dataset` | Obtener un dataset por su ID |
| `search_datasets_by_title` | Buscar datasets por texto en el título |
| `get_datasets_by_publisher` | Filtrar datasets por publicador/organismo |
| `get_datasets_by_theme` | Filtrar datasets por temática (economía, salud, etc.) |
| `get_datasets_by_format` | Filtrar datasets por formato (CSV, JSON, XML, etc.) |
| `get_datasets_by_keyword` | Filtrar datasets por palabra clave/etiqueta |
| `get_datasets_by_spatial` | Filtrar datasets por ámbito geográfico |
| `get_datasets_by_date_range` | Filtrar datasets modificados en un rango de fechas |

### Distribuciones (3 herramientas)

| Herramienta | Descripción |
|-------------|-------------|
| `list_distributions` | Listar todas las distribuciones (archivos) |
| `get_distributions_by_dataset` | Obtener archivos descargables de un dataset |
| `get_distributions_by_format` | Filtrar distribuciones por formato |

### Metadatos (3 herramientas)

| Herramienta | Descripción |
|-------------|-------------|
| `list_publishers` | Listar todos los publicadores (organismos) |
| `list_themes` | Listar todas las temáticas/categorías |
| `list_spatial_coverage` | Listar opciones de cobertura geográfica |

### NTI - Norma Técnica de Interoperabilidad (7 herramientas)

| Herramienta | Descripción |
|-------------|-------------|
| `list_public_sectors` | Listar sectores públicos |
| `get_public_sector` | Obtener un sector público por ID |
| `list_provinces` | Listar provincias españolas |
| `get_province` | Obtener una provincia por nombre |
| `list_autonomous_regions` | Listar Comunidades Autónomas |
| `get_autonomous_region` | Obtener una Comunidad Autónoma por ID |
| `get_country_spain` | Obtener información de España |

## Ejemplos de Uso

### Buscar datasets sobre empleo

```
Usuario: Busca datasets relacionados con empleo
Asistente: [Usa search_datasets_by_title("empleo")]
```

### Filtrar por temática

```
Usuario: ¿Qué datos hay disponibles sobre economía?
Asistente: [Usa get_datasets_by_theme("economia")]
```

### Obtener archivos de un dataset

```
Usuario: Dame los archivos descargables del dataset de presupuestos
Asistente: [Usa get_distributions_by_dataset("id-del-dataset")]
```

### Listar datasets en formato CSV

```
Usuario: Quiero ver qué datasets tienen datos en CSV
Asistente: [Usa get_datasets_by_format("csv")]
```

## Configuración en Clientes MCP

### Claude Desktop

Añade a tu archivo de configuración `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "datos-gob-es": {
      "command": "mcp",
      "args": ["run", "/ruta/a/datos-gob-es-mcp/src/datos_gob_es_mcp/server.py"]
    }
  }
}
```

### Cursor / VS Code

Añade a la configuración MCP del editor:

```json
{
  "mcp.servers": {
    "datos-gob-es": {
      "command": "mcp",
      "args": ["run", "/ruta/a/datos-gob-es-mcp/src/datos_gob_es_mcp/server.py"]
    }
  }
}
```

## Desarrollo

### Comandos disponibles

```bash
make help          # Mostrar ayuda
make dev           # Instalar en modo desarrollo
make run           # Ejecutar servidor
make run-stdio     # Ejecutar en modo stdio
make inspect       # Inspeccionar herramientas MCP
make test          # Ejecutar tests
make lint          # Verificar código con ruff
make format        # Formatear código con ruff
make clean         # Limpiar archivos de caché
make build         # Construir paquete
```

### Estructura del proyecto

```
datos-gob-es-mcp/
├── src/
│   └── datos_gob_es_mcp/
│       ├── __init__.py       # Exports y versión
│       └── server.py         # Servidor MCP completo (single file)
├── pyproject.toml            # Configuración del paquete
├── Makefile                  # Comandos de desarrollo
├── MANUAL_API_DATOS.md       # Documentación de la API
└── README.md
```

## API de datos.gob.es

Este servidor consume la API REST de datos.gob.es, que proporciona acceso al catálogo de datos abiertos del Gobierno de España.

- **Base URL**: `https://datos.gob.es/apidata/`
- **Documentación oficial**: [datos.gob.es/apidata](https://datos.gob.es/es/accessible-apidata)
- **Formatos soportados**: JSON, XML, RDF, Turtle, CSV

### Parámetros de paginación

- `_page`: Número de página (empieza en 0)
- `_pageSize`: Tamaño de página (máximo 50)
- `_sort`: Campo de ordenación (prefijo `-` para descendente)

## Despliegue en FastMCP Cloud

El servidor está preparado para desplegarse en FastMCP Cloud:

```bash
# Construir el paquete
make build

# El entry point está configurado en pyproject.toml
# datos-gob-es-mcp = "datos_gob_es_mcp.server:main"
```

## Licencia

MIT License - ver [LICENSE](LICENSE) para más detalles.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request en el repositorio.

## Enlaces

- [datos.gob.es](https://datos.gob.es/) - Portal de datos abiertos del Gobierno de España
- [Model Context Protocol](https://modelcontextprotocol.io/) - Especificación MCP
- [FastMCP](https://github.com/jlowin/fastmcp) - Framework para servidores MCP
