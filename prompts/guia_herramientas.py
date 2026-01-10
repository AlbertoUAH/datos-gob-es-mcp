"""Prompt for interactive tools documentation with examples."""

PROMPT_NAME = "guia_herramientas"
PROMPT_DESCRIPTION = """Guia interactiva de herramientas MCP disponibles.

Este prompt proporciona documentacion completa de todas las herramientas
disponibles con ejemplos de uso practicos."""


def generate_prompt(
    tool_category: str = "all",
    include_examples: bool = True
) -> str:
    """
    Generate the tools documentation prompt.

    Args:
        tool_category: Category to show ('all', 'search', 'external', 'utilities')
        include_examples: Whether to include usage examples
    """
    examples_section = ""
    if include_examples:
        examples_section = """
## Ejemplos Practicos

### Ejemplo 1: Buscar y descargar datos de presupuestos
```
1. search(title="presupuesto", theme="hacienda")
2. get(dataset_id="...", include_data=true, format="csv")
```

### Ejemplo 2: Busqueda semantica de datos
```
1. search(query="datos de desempleo juvenil en Madrid")
   -> Usa IA para encontrar datasets relevantes
```

### Ejemplo 3: Busqueda multi-tema
```
1. search(themes=["economia", "empleo"], fetch_all=true, max_results=100)
```

### Ejemplo 4: Obtener estadisticas del INE
```
1. ine_search(query="empleo")  -> Encontrar operacion (ej: EPA, id=30308)
2. ine_search(operation_id="30308")  -> Listar tablas
3. ine_download(table_id="4247", n_last=12)  -> Descargar datos
```
"""

    search_tools = """
### Herramientas de Busqueda

**search** - Busqueda unificada de datasets
- `query`: Busqueda en lenguaje natural (usa IA)
- `title`: Buscar en titulos
- `keyword`: Buscar por palabra clave
- `theme` / `themes`: Filtrar por tematica(s)
- `publisher`: Filtrar por publicador (IDs en instrucciones del servidor)
- `format`: Filtrar por formato (csv, json, xml)
- `fetch_all`: Obtener todos los resultados (paginacion automatica)
- `include_preview`: Incluir vista previa de datos

**get** - Obtener dataset con opcion de descarga
- `dataset_id`: ID del dataset
- `include_data`: Si true, descarga datos (default: false)
- `format`: Formato preferido cuando include_data=true
- `max_rows`: Limite de filas
- `max_mb`: Limite de tamano (default 10, max 50)
- `lang`: Idioma preferido (es, en, ca, eu, gl)
"""

    external_tools = """
### Integraciones Externas

#### INE (Instituto Nacional de Estadistica)
- `ine_search`: Buscar operaciones o listar tablas
  - `query`: Buscar operaciones (empleo, IPC, poblacion...)
  - `operation_id`: Si se indica, lista tablas de esa operacion
- `ine_download`: Descargar datos de una tabla
  - `table_id`: ID de tabla
  - `n_last`: Ultimos N periodos (default 10)

#### AEMET (Meteorologia)
- `aemet_list_locations`: Listar municipios y/o estaciones
  - `location_type`: 'municipalities', 'stations', o 'all'
- `aemet_get_forecast`: Prediccion para un municipio
  - `municipality_code`: Codigo de 5 digitos (ej: 28079 Madrid)
- `aemet_get_observations`: Observaciones actuales
  - `station_id`: ID de estacion (opcional)

#### BOE (Boletin Oficial del Estado)
- `boe_get_summary`: Sumario del BOE
  - `date`: Fecha YYYYMMDD (o None para el mas reciente)
- `boe_get_document`: Obtener documento por ID
  - `document_id`: ID del documento (ej: BOE-A-2024-12345)
- `boe_search`: Buscar documentos
  - `query`: Texto de busqueda
  - `date_from`, `date_to`: Rango de fechas
"""

    # Build content based on category
    if tool_category == "search":
        tools_content = search_tools
    elif tool_category == "external":
        tools_content = external_tools
    else:
        tools_content = search_tools + external_tools

    return f"""# Guia de Herramientas MCP - datos.gob.es

## Descripcion General

Este servidor MCP proporciona acceso al catalogo de datos abiertos de Espana
(datos.gob.es) con herramientas organizadas en categorias:

- **Busqueda**: Buscar y descargar datasets (search, get)
- **INE**: Estadisticas oficiales de Espana
- **AEMET**: Datos meteorologicos
- **BOE**: Legislacion y normativa

## Herramientas Disponibles
{tools_content}
{examples_section}
## Recursos MCP Disponibles

### Resource Templates (acceso directo por URI)
- `dataset://{{id}}` - Info de un dataset
- `theme://{{id}}` - Datasets de una tematica
- `publisher://{{id}}` - Datasets de un publicador
- `format://{{id}}` - Datasets en un formato
- `keyword://{{palabra}}` - Datasets con palabra clave

## Referencia Rapida de IDs

### Temas (usar con theme=)
economia, hacienda, educacion, salud, medio-ambiente, transporte,
turismo, empleo, sector-publico, ciencia-tecnologia, cultura-ocio,
urbanismo-infraestructuras, energia

### Publicadores principales (usar con publisher=)
- EA0010587: INE
- E05024401: Min. Hacienda
- E00003901: AEMET
- L01280796: Ayto. Madrid
- L01080193: Ayto. Barcelona

## Consejos de Uso

1. **Busqueda semantica**: Usa `query` para busquedas en lenguaje natural.
2. **Descarga integrada**: `get(id, include_data=true)` en una sola llamada.
3. **Multi-tema**: Usa `themes=["economia", "empleo"]` para buscar en varios.
4. **INE paso a paso**: ine_search(query) -> ine_search(operation_id) -> ine_download(table_id)
"""
