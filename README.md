# datos-gob-es-mcp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

Servidor MCP (Model Context Protocol) para acceder al catálogo de datos abiertos de España a través de la API de **datos.gob.es**.

## Descripción

Este servidor MCP permite a asistentes de IA como Claude, ChatGPT y otros clientes compatibles con MCP buscar y explorar los miles de datasets públicos disponibles en el portal de datos abiertos del Gobierno de España.

### Características

- **22 herramientas MCP** para consultar la API de datos.gob.es
- **13 recursos MCP** (9 estáticos + 4 templates dinámicos) para acceso directo a datos
- **4 prompts MCP** para guías de búsqueda detalladas
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

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Ejecutar el servidor MCP

```bash
# Modo stdio (para clientes MCP)
make run-stdio

# O directamente
mcp run server.py
```

### Inspeccionar herramientas disponibles

```bash
make inspect
```

## Capacidades MCP

Este servidor implementa las tres capacidades principales del Model Context Protocol:

| Capacidad | Cantidad | Descripción |
|-----------|----------|-------------|
| **Tools** | 22 | Funciones que el LLM puede invocar |
| **Resources** | 13 | Datos estáticos y dinámicos accesibles |
| **Prompts** | 4 | Guías de búsqueda predefinidas |

---

## Tools (Herramientas)

Las herramientas permiten al asistente de IA realizar operaciones activas sobre la API.

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

---

## Resources (Recursos)

Los recursos proporcionan acceso directo a datos sin necesidad de invocar herramientas.

### Recursos Estáticos (9)

| URI | Descripción |
|-----|-------------|
| `catalog://overview` | Resumen general del catálogo con estadísticas |
| `catalog://themes` | Lista de todas las temáticas disponibles |
| `catalog://publishers` | Lista de todos los organismos publicadores |
| `catalog://formats` | Formatos de datos disponibles (CSV, JSON, XML, etc.) |
| `catalog://spatial` | Opciones de cobertura geográfica |
| `catalog://provinces` | Lista de provincias españolas |
| `catalog://autonomous-regions` | Lista de Comunidades Autónomas |
| `catalog://public-sectors` | Sectores públicos según NTI |
| `catalog://keywords` | Palabras clave más utilizadas |

### Resource Templates Dinámicos (4)

| URI Template | Descripción | Ejemplo |
|--------------|-------------|---------|
| `dataset://{dataset_id}` | Información completa de un dataset | `dataset://l01280066-presupuestos-2024` |
| `theme://{theme_id}` | Datasets de una temática específica | `theme://economia` |
| `publisher://{publisher_id}` | Datasets de un publicador específico | `publisher://E00003901` |
| `format://{format_id}` | Datasets en un formato específico | `format://csv` |
| `keyword://{keyword}` | Datasets con una palabra clave | `keyword://presupuestos` |

### Uso de Resources

Los resources se pueden usar directamente en conversaciones:

```
Usuario: Dame información sobre el catálogo
Asistente: [Lee catalog://overview]

Usuario: ¿Qué datasets hay sobre economía?
Asistente: [Lee theme://economia]

Usuario: Muéstrame el dataset l01280066-presupuestos
Asistente: [Lee dataset://l01280066-presupuestos]
```

---

## Prompts (Guías de Búsqueda)

Los prompts proporcionan guías detalladas para tareas complejas de búsqueda y análisis.

### Prompts Disponibles (4)

| Prompt | Parámetros | Descripción |
|--------|------------|-------------|
| `prompt_buscar_datos_por_tema` | tema, formato, max_resultados | Guía para encontrar datasets de una temática en un formato específico |
| `prompt_datasets_recientes` | dias, tema, max_resultados | Guía para encontrar datasets actualizados recientemente |
| `prompt_explorar_catalogo` | interes | Guía completa de exploración del catálogo |
| `prompt_analisis_dataset` | dataset_id, incluir_distribuciones, evaluar_calidad | Guía para análisis detallado de un dataset |

### Ejemplo: Buscar datos por tema

```
Prompt: prompt_buscar_datos_por_tema(tema="salud", formato="csv", max_resultados=10)

El prompt guiará al asistente para:
1. Verificar la temática en catalog://themes
2. Buscar datasets con get_datasets_by_theme
3. Filtrar por formato CSV
4. Obtener detalles de los datasets más relevantes
```

### Ejemplo: Analizar un dataset

```
Prompt: prompt_analisis_dataset(dataset_id="l01280066-presupuestos", evaluar_calidad=true)

El prompt guiará al asistente para:
1. Obtener metadatos completos del dataset
2. Analizar las distribuciones disponibles
3. Evaluar la calidad de los datos
4. Identificar posibles casos de uso
5. Sugerir datasets complementarios
```

---

## Ejemplos de Uso

### Buscar datasets sobre empleo

```
Usuario: Busca datasets relacionados con empleo
Asistente: [Usa search_datasets_by_title("empleo")]
```

### Filtrar por temática

```
Usuario: ¿Qué datos hay disponibles sobre economía?
Asistente: [Lee theme://economia o usa get_datasets_by_theme("economia")]
```

### Obtener archivos de un dataset

```
Usuario: Dame los archivos descargables del dataset de presupuestos
Asistente: [Usa get_distributions_by_dataset("id-del-dataset")]
```

### Exploración guiada

```
Usuario: Quiero explorar datos sobre turismo en España
Asistente: [Usa prompt_explorar_catalogo(interes="turismo en España")]
```

### Análisis de dataset

```
Usuario: Analiza el dataset l01280066-presupuestos-2024
Asistente: [Usa prompt_analisis_dataset(dataset_id="l01280066-presupuestos-2024")]
```

## Configuración en Clientes MCP

### Claude Desktop

Añade a tu archivo de configuración `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "datos-gob-es": {
      "command": "mcp",
      "args": ["run", "/ruta/a/datos-gob-es-mcp/server.py"]
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
      "args": ["run", "/ruta/a/datos-gob-es-mcp/server.py"]
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
```

### Estructura del proyecto

```
datos-gob-es-mcp/
├── server.py                 # Servidor MCP (tools + resources)
├── prompts/                  # Guías de búsqueda MCP
│   ├── __init__.py          # Registro de prompts
│   ├── buscar_por_tema.py   # Búsqueda por temática y formato
│   ├── datasets_recientes.py # Datasets actualizados recientemente
│   ├── explorar_catalogo.py # Exploración guiada del catálogo
│   └── analisis_dataset.py  # Análisis detallado de datasets
├── requirements.txt          # Dependencias Python
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

El servidor está preparado para desplegarse en FastMCP Cloud. El archivo principal es `server.py` en la raíz del proyecto.

```bash
# Entry point para FastMCP Cloud
server.py
```

## Licencia

MIT License - ver [LICENSE](LICENSE) para más detalles.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request en el repositorio.

## Enlaces

- [datos.gob.es](https://datos.gob.es/) - Portal de datos abiertos del Gobierno de España
- [Model Context Protocol](https://modelcontextprotocol.io/) - Especificación MCP
- [FastMCP](https://github.com/jlowin/fastmcp) - Framework para servidores MCP
