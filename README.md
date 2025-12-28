# datos-gob-es-mcp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

Servidor MCP (Model Context Protocol) para acceder al catalogo de datos abiertos de Espana a traves de la API de **datos.gob.es**, con integraciones adicionales para **INE**, **AEMET** y **BOE**.

## Descripcion

Este servidor MCP permite a asistentes de IA como Claude, ChatGPT y otros clientes compatibles con MCP buscar y explorar los miles de datasets publicos disponibles en el portal de datos abiertos del Gobierno de Espana.

### Caracteristicas

- **32 herramientas MCP** para consultar multiples APIs de datos publicos
- **13 recursos MCP** (9 estaticos + 4 templates dinamicos) para acceso directo a datos
- **5 prompts MCP** para guias de busqueda detalladas
- **Integraciones externas**:
  - **INE**: Estadisticas oficiales de Espana
  - **AEMET**: Datos meteorologicos (requiere API key gratuita)
  - **BOE**: Boletin Oficial del Estado
- **Sistema de notificaciones**: Webhooks para detectar cambios en datasets
- **Busqueda semantica**: Busqueda por significado usando embeddings
- **Cache de metadatos**: Cache local de 24h para publishers, themes, provincias y regiones
- **Paginacion paralela**: Descarga 5x mas rapida con `fetch_all=True`
- **Descarga completa**: Tool `download_data` para obtener datasets completos (hasta 50MB)
- Cliente HTTP asincrono con manejo robusto de errores
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
| `WEBHOOK_SECRET` | No | Secreto para validar firmas de webhooks |
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

## Capacidades MCP

| Capacidad | Cantidad | Descripcion |
|-----------|----------|-------------|
| **Tools** | 32 | Funciones que el LLM puede invocar |
| **Resources** | 13 | Datos estaticos y dinamicos accesibles |
| **Prompts** | 5 | Guias de busqueda predefinidas |

---

## Tools (Herramientas)

### Datasets - datos.gob.es (5 herramientas principales)

| Herramienta | Descripcion |
|-------------|-------------|
| `search_datasets` | Busqueda unificada: filtros, semantica e hibrida. Soporta multi-tema con logica OR |
| `get_dataset` | Obtener un dataset por su ID |
| `download_data` | **Nuevo**: Descargar datos completos de un dataset (hasta 50MB) |
| `get_related_datasets` | **Nuevo**: Encontrar datasets similares usando IA (embeddings) |
| `get_distributions` | Obtener archivos descargables de un dataset |

### Metadatos (3 herramientas con cache 24h)

| Herramienta | Descripcion |
|-------------|-------------|
| `list_publishers` | Listar todos los publicadores (organismos). Cache 24h |
| `list_themes` | Listar todas las tematicas/categorias. Cache 24h |
| `refresh_metadata_cache` | **Nuevo**: Forzar actualizacion del cache de metadatos |

### NTI - Norma Tecnica de Interoperabilidad (3 herramientas con cache 24h)

| Herramienta | Descripcion |
|-------------|-------------|
| `list_public_sectors` | Listar sectores publicos. Cache 24h |
| `list_provinces` | Listar provincias espanolas. Cache 24h |
| `list_autonomous_regions` | Listar Comunidades Autonomas. Cache 24h |

### Integraciones Externas

#### INE - Instituto Nacional de Estadistica (4 herramientas)

| Herramienta | Descripcion |
|-------------|-------------|
| `ine_list_operations` | Listar operaciones estadisticas |
| `ine_search_operations` | Buscar operaciones por texto |
| `ine_list_tables` | Listar tablas de una operacion |
| `ine_get_data` | Obtener datos de una tabla |

#### AEMET - Meteorologia (4 herramientas)

| Herramienta | Descripcion |
|-------------|-------------|
| `aemet_list_stations` | Listar estaciones meteorologicas |
| `aemet_list_municipalities` | Listar municipios |
| `aemet_get_observations` | Obtener observaciones de una estacion |
| `aemet_get_forecast` | Obtener prediccion para un municipio |

#### BOE - Boletin Oficial del Estado (4 herramientas)

| Herramienta | Descripcion |
|-------------|-------------|
| `boe_get_today` | Obtener sumario del BOE de hoy |
| `boe_get_summary` | Obtener sumario de una fecha |
| `boe_get_document` | Obtener documento por ID |
| `boe_search` | Buscar documentos |

#### Webhooks - Notificaciones (5 herramientas)

| Herramienta | Descripcion |
|-------------|-------------|
| `webhook_register` | Registrar webhook para un dataset |
| `webhook_list` | Listar webhooks registrados |
| `webhook_delete` | Eliminar webhook |
| `webhook_test` | Probar webhook |
| `check_dataset_changes` | Verificar cambios en datasets vigilados |
| `list_watched_datasets` | Listar datasets vigilados |

#### Utilidades (3 herramientas)

| Herramienta | Descripcion |
|-------------|-------------|
| `export_results` | **Nuevo**: Exportar resultados de busqueda a CSV o JSON |
| `get_usage_stats` | **Nuevo**: Ver estadisticas de uso de herramientas y datasets |
| `clear_usage_stats` | **Nuevo**: Limpiar estadisticas de uso |

---

## Resources (Recursos)

### Recursos Estaticos

| URI | Descripcion |
|-----|-------------|
| `catalog://themes` | Lista de todas las tematicas disponibles |
| `catalog://publishers` | Lista de todos los organismos publicadores |
| `catalog://provinces` | Lista de provincias espanolas |
| `catalog://autonomous-regions` | Lista de Comunidades Autonomas |

### Resource Templates Dinamicos

| URI Template | Descripcion | Ejemplo |
|--------------|-------------|---------|
| `dataset://{dataset_id}` | Informacion de un dataset | `dataset://l01280066-presupuestos` |
| `theme://{theme_id}` | Datasets de una tematica | `theme://economia` |
| `publisher://{publisher_id}` | Datasets de un publicador | `publisher://E00003901` |
| `format://{format_id}` | Datasets en un formato | `format://csv` |
| `keyword://{keyword}` | Datasets con una palabra clave | `keyword://presupuestos` |

---

## Ejemplos de Uso

### Buscar datasets por texto

```
Usuario: Busca datasets sobre empleo en Andalucia
Asistente: [Usa search_datasets(title="empleo", spatial_type="Autonomia", spatial_value="Andalucia")]
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

### Obtener distribuciones de un dataset

```
Usuario: Que formatos tiene disponible este dataset?
Asistente: [Usa get_distributions(dataset_id="l01280066-presupuestos")]
```

### Exportar resultados de busqueda (nuevo)

```
Usuario: Exporta estos resultados a CSV
Asistente: [Usa export_results(search_results="...", format="csv")]
```

### Ver estadisticas de uso (nuevo)

```
Usuario: Que herramientas he usado mas?
Asistente: [Usa get_usage_stats(include_searches=true)]
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
├── notifications/            # Sistema de webhooks
│   ├── webhook.py           # Gestor de webhooks
│   └── watcher.py           # Vigilante de cambios
├── prompts/                  # Guias de busqueda MCP
├── examples/                 # Jupyter notebooks de ejemplo
│   ├── 01_introduccion.ipynb
│   ├── 02_busqueda_datasets.ipynb
│   ├── 03_analisis_datos.ipynb
│   ├── 04_integraciones.ipynb
│   └── 05_webhooks.ipynb
├── tests/                    # Tests automatizados
├── docs/                     # Documentacion adicional
│   └── DEPLOYMENT.md        # Guia de despliegue
├── requirements.txt         # Dependencias Python
├── Makefile                 # Comandos de desarrollo
└── README.md
```

## APIs Integradas

| API | Autenticacion | Documentacion |
|-----|---------------|---------------|
| datos.gob.es | No | [datos.gob.es/apidata](https://datos.gob.es/es/accessible-apidata) |
| INE | No | [ine.es/dyngs/DataLab](https://www.ine.es/dyngs/DataLab/es/manual.html) |
| AEMET | Si (gratis) | [opendata.aemet.es](https://opendata.aemet.es/dist/index.html) |
| BOE | No | [boe.es/datosabiertos](https://www.boe.es/datosabiertos/) |

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
| **Filtro espacial mejorado** | Busca en campo spatial, titulo y URI | Encuentra datasets sin metadatos espaciales |

## Licencia

MIT License - ver [LICENSE](LICENSE) para mas detalles.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request en el repositorio.

## Enlaces

- [datos.gob.es](https://datos.gob.es/) - Portal de datos abiertos del Gobierno de Espana
- [Model Context Protocol](https://modelcontextprotocol.io/) - Especificacion MCP
- [FastMCP](https://github.com/jlowin/fastmcp) - Framework para servidores MCP
- [INE](https://www.ine.es/) - Instituto Nacional de Estadistica
- [AEMET OpenData](https://opendata.aemet.es/) - Datos meteorologicos abiertos
- [BOE](https://www.boe.es/) - Boletin Oficial del Estado
