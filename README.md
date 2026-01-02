# datos-gob-es-mcp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

**Hub de OpenData Espanol** - Servidor MCP (Model Context Protocol) que unifica el acceso a las principales fuentes de datos abiertos de Espana en una sola interfaz.

## Descripcion

Este servidor MCP actua como un **hub centralizado** que conecta multiples APIs de datos publicos espanoles, permitiendo a asistentes de IA como Claude, ChatGPT y otros clientes MCP acceder a toda la informacion desde un unico punto.

### Fuentes de datos integradas

| Fuente | Descripcion | Auth | Documentacion |
|--------|-------------|------|---------------|
| **[datos.gob.es](https://datos.gob.es/)** | Catalogo nacional de datos abiertos (+40.000 datasets) | No | [API](https://datos.gob.es/es/accessible-apidata) |
| **[INE](https://www.ine.es/)** | Instituto Nacional de Estadistica | No | [API](https://www.ine.es/dyngs/DataLab/es/manual.html) |
| **[AEMET](https://opendata.aemet.es/)** | Agencia Estatal de Meteorologia | API key | [API](https://opendata.aemet.es/dist/index.html) |
| **[BOE](https://www.boe.es/)** | Boletin Oficial del Estado | No | [API](https://www.boe.es/datosabiertos/) |

### Caracteristicas

- **18 herramientas MCP** para consultar multiples APIs de datos publicos
- **5 recursos MCP** (templates dinamicos) para acceso directo a datos
- **6 prompts MCP** para guias de busqueda detalladas
- **Busqueda semantica**: Busqueda por significado usando embeddings (IA)
- **Busqueda espacial inteligente**: Filtra por ubicacion usando metadatos, titulos y URIs
- **Cache de metadatos**: Cache local de 24h para respuestas instantaneas
- **Paginacion paralela**: Descarga 5x mas rapida con `fetch_all=True`
- **Descarga completa**: Obtener datasets completos (hasta 50MB)
- **Exportacion**: Exportar resultados a CSV/JSON
- Cliente HTTP asincrono con rate limiting por API
- Modelos Pydantic para tipado seguro
- Listo para desplegar en FastMCP Cloud

## Instalacion

### Requisitos

- Python 3.10 o superior
- pip

### Instalacion rapida

```bash
# Clonar el repositorio
git clone https://github.com/AlbertoUAH/datos-gob-es-mcp.git
cd datos-gob-es-mcp

# Crear entorno virtual e instalar
make dev
```

### Instalacion manual

```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Configuracion

### Variables de entorno

Crea un archivo `.env` basandote en `.env.example`:

```bash
cp .env.example .env
```

| Variable | Requerida | Descripcion |
|----------|-----------|-------------|
| `AEMET_API_KEY` | Para meteorologia | API key de AEMET OpenData ([obtener gratis](https://opendata.aemet.es/centrodedescargas/altaUsuario)) |
| `LOG_LEVEL` | No | Nivel de logging: DEBUG, INFO, WARNING, ERROR (default: INFO) |
| `LOG_FORMAT` | No | Formato de logs: console o json (default: console) |
| `RATE_LIMIT_DATOS_GOB_ES` | No | Peticiones/segundo a datos.gob.es (default: 10) |
| `RATE_LIMIT_INE` | No | Peticiones/segundo a INE (default: 5) |
| `RATE_LIMIT_AEMET` | No | Peticiones/segundo a AEMET (default: 10) |
| `RATE_LIMIT_BOE` | No | Peticiones/segundo a BOE (default: 10) |

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

## Arquitectura

```mermaid
flowchart TB
    subgraph Cliente["Cliente MCP"]
        ChatGPT["ChatGPT"]
    end

    subgraph MCP["Servidor MCP (FastMCP)"]
        Server["server.py"]
    end

    ChatGPT <-->|"Protocolo MCP"| Server

    subgraph Tools["TOOLS (18)"]
        subgraph ToolsDatosGob["datos.gob.es (5)"]
            search_datasets
            get_dataset
            download_data
            get_related_datasets
            list_metadata
        end

        subgraph ToolsINE["INE (3)"]
            ine_list_operations
            ine_list_tables
            ine_get_data
        end

        subgraph ToolsAEMET["AEMET (4)"]
            aemet_list_stations
            aemet_list_municipalities
            aemet_get_observations
            aemet_get_forecast
        end

        subgraph ToolsBOE["BOE (3)"]
            boe_get_summary
            boe_get_document
            boe_search
        end

        subgraph ToolsUtils["Utilidades (3)"]
            export_results
            get_usage_stats
            refresh_metadata_cache
        end
    end

    subgraph Resources["RESOURCES (5)"]
        R1["dataset://{id}"]
        R2["theme://{id}"]
        R3["publisher://{id}"]
        R4["format://{id}"]
        R5["keyword://{keyword}"]
    end

    subgraph Prompts["PROMPTS (6)"]
        P1["buscar_datos_por_tema"]
        P2["datasets_recientes"]
        P3["explorar_catalogo"]
        P4["analisis_dataset"]
        P5["guia_herramientas"]
        P6["buscar_estadisticas"]
    end

    Server --> Tools
    Server --> Resources
    Server --> Prompts

    subgraph APIs["APIs Externas"]
        API1["datos.gob.es"]
        API2["INE"]
        API3["AEMET"]
        API4["BOE"]
    end

    ToolsDatosGob --> API1
    Resources --> API1
    ToolsINE --> API2
    ToolsAEMET --> API3
    ToolsBOE --> API4
```

## Capacidades MCP

| Capacidad | Cantidad | Descripcion |
|-----------|----------|-------------|
| **Tools** | 18 | Funciones que el LLM puede invocar |
| **Resources** | 5 | Templates dinamicos para acceso directo |
| **Prompts** | 6 | Guias de busqueda predefinidas |

---

## Tools (Herramientas)

### datos.gob.es (5 herramientas)

| Herramienta | Descripcion |
|-------------|-------------|
| `search_datasets` | Busqueda unificada de datasets: por filtros (titulo, tema, publicador, formato, fecha), semantica (IA con embeddings) o hibrida. Soporta multi-tema con logica OR y paginacion paralela |
| `get_dataset` | Obtiene metadatos completos de un dataset por su ID: titulo, descripcion, publicador, frecuencia de actualizacion, formatos disponibles y distribuciones |
| `download_data` | Descarga y parsea datos de un dataset (hasta 50MB). Soporta CSV y JSON, devuelve preview de filas |
| `get_related_datasets` | Encuentra datasets similares usando busqueda semantica con embeddings de IA |
| `list_metadata` | Lista metadatos del catalogo: publicadores (`publishers`), tematicas (`themes`), sectores publicos (`public_sectors`), provincias (`provinces`) o comunidades autonomas (`autonomous_regions`). Cache de 24h |

### INE - Instituto Nacional de Estadistica (3 herramientas) - FUENTE PRINCIPAL DE ESTADISTICAS

El INE es la **fuente oficial principal** de estadisticas en Espana. Contiene datos de empleo (EPA), poblacion, precios (IPC), PIB, turismo, censos, y mas.

| Herramienta | Descripcion |
|-------------|-------------|
| `ine_list_operations` | Busca operaciones estadisticas del INE (empleo, poblacion, IPC, PIB, turismo, etc.). Usa `query` para filtrar |
| `ine_list_tables` | Lista las tablas disponibles para una operacion estadistica |
| `ine_get_data` | Obtiene datos estadisticos reales de una tabla del INE |

### AEMET - Meteorologia (4 herramientas)

| Herramienta | Descripcion |
|-------------|-------------|
| `aemet_list_stations` | Lista todas las estaciones meteorologicas de AEMET con su ubicacion |
| `aemet_list_municipalities` | Lista los municipios espanoles con codigos para predicciones |
| `aemet_get_observations` | Obtiene observaciones meteorologicas actuales de una estacion |
| `aemet_get_forecast` | Obtiene la prediccion meteorologica para un municipio |

### BOE - Boletin Oficial del Estado (3 herramientas)

| Herramienta | Descripcion |
|-------------|-------------|
| `boe_get_summary` | Obtiene el sumario del BOE. Si no se especifica fecha, devuelve el BOE mas reciente (hoy o hasta 7 dias atras) |
| `boe_get_document` | Obtiene metadatos completos de un documento del BOE por su ID (ej: `BOE-A-2024-12345`) |
| `boe_search` | Busca documentos en el BOE por texto en un rango de fechas (hasta 90 dias) |

### Utilidades (3 herramientas)

| Herramienta | Descripcion |
|-------------|-------------|
| `export_results` | Exporta resultados de busqueda a formato CSV o JSON |
| `get_usage_stats` | Muestra estadisticas de uso: tools mas usadas, datasets mas accedidos |
| `refresh_metadata_cache` | Fuerza la actualizacion del cache de metadatos |

---

## Resources (Recursos)

Templates dinamicos para acceso directo a datos de datos.gob.es:

| URI Template | Descripcion | Ejemplo |
|--------------|-------------|---------|
| `dataset://{dataset_id}` | Informacion de un dataset | `dataset://l01280066-presupuestos` |
| `theme://{theme_id}` | Datasets de una tematica | `theme://economia` |
| `publisher://{publisher_id}` | Datasets de un publicador | `publisher://E00003901` |
| `format://{format_id}` | Datasets en un formato | `format://csv` |
| `keyword://{keyword}` | Datasets con una palabra clave | `keyword://presupuestos` |

---

## Prompts (Guias de Busqueda)

Los prompts proporcionan guias estructuradas para tareas comunes de busqueda:

| Prompt | Descripcion |
|--------|-------------|
| `buscar_datos_por_tema` | Busqueda guiada de datasets por tematica y formato |
| `datasets_recientes` | Encontrar datasets actualizados en los ultimos dias |
| `explorar_catalogo` | Exploracion guiada del catalogo de datos abiertos |
| `analisis_dataset` | Analisis detallado de un dataset especifico |
| `guia_herramientas` | Documentacion de todas las herramientas MCP |
| `buscar_estadisticas` | **Busqueda de estadisticas oficiales** consultando INE y datos.gob.es |

### Prompt destacado: buscar_estadisticas

Este prompt guia al asistente para buscar datos estadisticos en **ambas fuentes**:

1. **INE (Instituto Nacional de Estadistica)**: Fuente PRINCIPAL de estadisticas oficiales espanolas (empleo, poblacion, precios, PIB, turismo, etc.)
2. **datos.gob.es**: Complementa con datasets de otros organismos

```
Usuario: Busca estadisticas de empleo en Espana
Asistente: [Usa prompt_buscar_estadisticas(tema="empleo")]
           -> Consulta INE (EPA, paro) + datos.gob.es
```

---

## Ejemplos de Uso

### Buscar datasets por texto

```
Usuario: Busca datasets sobre empleo en Andalucia
Asistente: [Usa search_datasets(title="empleo Andalucia")]
```

### Buscar por significado (semantica)

```
Usuario: Encuentra datos sobre desempleo juvenil
Asistente: [Usa search_datasets(semantic_query="desempleo juvenil")]
```

### Buscar por multiples temas (nuevo)

```
Usuario: Busca datasets de economia o hacienda
Asistente: [Usa search_datasets(themes=["economia", "hacienda"])]
```

### Descargar datos completos (nuevo)

```
Usuario: Descarga los datos del dataset de presupuestos
Asistente: [Usa download_data(dataset_id="l01280066-presupuestos", max_mb=20)]
```

### Encontrar datasets relacionados (nuevo)

```
Usuario: Encuentra datasets similares a este de poblacion
Asistente: [Usa get_related_datasets(dataset_id="...", top_k=10)]
```

### Exportar resultados de busqueda (nuevo)

```
Usuario: Exporta estos resultados a CSV
Asistente: [Usa export_results(search_results="...", format="csv")]
```

### Ver estadisticas de uso

```
Usuario: Que herramientas he usado mas?
Asistente: [Usa get_usage_stats(include_searches=true)]
```

### Listar metadatos del catalogo

```
Usuario: Muestra las tematicas disponibles
Asistente: [Usa list_metadata(metadata_type="themes")]

Usuario: Lista las provincias espanolas
Asistente: [Usa list_metadata(metadata_type="provinces")]
```

### Buscar operaciones del INE

```
Usuario: Busca estadisticas sobre empleo
Asistente: [Usa ine_list_operations(query="empleo")]
```

### Obtener el BOE mas reciente

```
Usuario: Dame el BOE de hoy
Asistente: [Usa boe_get_summary()]

Usuario: Dame el BOE del 2 de enero de 2025
Asistente: [Usa boe_get_summary(date="20250102")]
```

## Configuracion en Clientes MCP

### Claude Desktop

Anade a tu archivo de configuracion `claude_desktop_config.json`:

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

## Desarrollo

### Comandos disponibles

```bash
make help          # Mostrar ayuda
make dev           # Instalar en modo desarrollo
make run           # Ejecutar servidor
make run-stdio     # Ejecutar en modo stdio
make inspect       # Inspeccionar herramientas MCP
make test          # Ejecutar tests
make lint          # Verificar codigo con ruff
make format        # Formatear codigo con ruff
make clean         # Limpiar archivos de cache
make notebooks     # Iniciar servidor Jupyter
```

### Estructura del proyecto

```
datos-gob-es-mcp/
├── server.py                 # Servidor MCP principal
├── core/                     # Modulo central
│   ├── logging.py           # Logging estructurado (structlog)
│   ├── ratelimit.py         # Rate limiting (aiolimiter)
│   └── http.py              # Cliente HTTP centralizado
├── integrations/             # APIs externas
│   ├── ine.py               # Instituto Nacional de Estadistica
│   ├── aemet.py             # Agencia de Meteorologia
│   └── boe.py               # Boletin Oficial del Estado
├── prompts/                  # Guias de busqueda MCP
├── examples/                 # Jupyter notebooks de ejemplo
│   ├── 01_introduccion.ipynb
│   ├── 02_busqueda_datasets.ipynb
│   ├── 03_analisis_datos.ipynb
│   └── 04_integraciones.ipynb
├── tests/                    # Tests automatizados
├── docs/                     # Documentacion adicional
│   └── DEPLOYMENT.md        # Guia de despliegue
├── requirements.txt         # Dependencias Python
├── Makefile                 # Comandos de desarrollo
└── README.md
```

## Rendimiento

### Optimizaciones implementadas

| Mejora | Descripcion | Impacto |
|--------|-------------|---------|
| **Cache de metadatos** | Publishers, themes, provincias y regiones se cachean 24h | Respuestas instantaneas en llamadas repetidas |
| **Paginacion paralela** | `fetch_all=True` descarga 5 paginas en paralelo | ~5x mas rapido |
| **Descarga streaming** | `download_data` usa streaming para archivos grandes | Soporte hasta 50MB |
| **Embeddings cacheados** | Indice semantico se guarda en disco | Primera busqueda ~30s, siguientes <1s |
| **Metricas de uso** | Registro de herramientas y datasets mas usados | Optimizacion de workflows |
| **Exportacion de resultados** | Exportar busquedas a CSV/JSON | Analisis externo de datos |

## Licencia

MIT License - ver [LICENSE](LICENSE) para mas detalles.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request en el repositorio.

## Enlaces

- [Model Context Protocol](https://modelcontextprotocol.io/) - Especificacion MCP
- [FastMCP](https://github.com/jlowin/fastmcp) - Framework para servidores MCP
