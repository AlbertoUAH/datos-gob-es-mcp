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
        tool_category: Category to show ('all', 'search', 'metadata', 'external', 'utilities')
        include_examples: Whether to include usage examples
    """
    examples_section = ""
    if include_examples:
        examples_section = """
## Ejemplos Practicos

### Ejemplo 1: Buscar y descargar datos de presupuestos
```
1. search_datasets(title="presupuesto", theme="hacienda", include_preview=true)
2. download_data(dataset_id="...", format="csv", max_mb=20)
3. export_results(search_results="...", format="csv")
```

### Ejemplo 2: Encontrar datasets similares
```
1. search_datasets(semantic_query="datos de desempleo juvenil")
2. get_related_datasets(dataset_id="...", top_k=5, min_score=0.6)
```

### Ejemplo 3: Busqueda multi-tema con exportacion
```
1. search_datasets(themes=["economia", "empleo"], fetch_all=true, max_results=100)
2. export_results(search_results="...", format="json")
```

### Ejemplo 4: Monitorizar uso de herramientas
```
1. get_usage_stats(include_searches=true)
2. clear_usage_stats()  # Para reiniciar metricas
```
"""

    search_tools = """
### Herramientas de Busqueda (search_datasets)

**search_datasets** - Busqueda unificada de datasets
- `title`: Buscar en titulos
- `keyword`: Buscar por palabra clave
- `theme` / `themes`: Filtrar por tematica(s)
- `publisher`: Filtrar por publicador
- `format`: Filtrar por formato (csv, json, xml)
- `spatial_type` + `spatial_value`: Filtrar por ubicacion
- `semantic_query`: Busqueda por significado con IA
- `fetch_all`: Obtener todos los resultados (paginacion automatica)
- `include_preview`: Incluir vista previa de datos

**get_dataset** - Obtener detalles de un dataset especifico
- `dataset_id`: ID del dataset
- `lang`: Idioma preferido (es, en, ca, eu, gl)

**get_distributions** - Obtener archivos descargables
- `dataset_id`: ID del dataset (opcional)
- `format`: Filtrar por formato

**download_data** - Descargar datos completos (hasta 50MB)
- `dataset_id`: ID del dataset
- `format`: Formato preferido
- `max_rows`: Limite de filas
- `max_mb`: Limite de tamano

**get_related_datasets** - Encontrar datasets similares con IA
- `dataset_id`: Dataset de referencia
- `top_k`: Numero maximo de resultados
- `min_score`: Puntuacion minima de similitud
"""

    metadata_tools = """
### Herramientas de Metadatos

**list_publishers** - Listar organismos publicadores
- `page`: Numero de pagina
- `use_cache`: Usar cache local (24h)

**list_themes** - Listar categorias tematicas
- `page`: Numero de pagina
- `use_cache`: Usar cache local

**list_provinces** - Listar provincias espanolas
**list_autonomous_regions** - Listar Comunidades Autonomas
**list_public_sectors** - Listar sectores publicos

**refresh_metadata_cache** - Refrescar cache de metadatos
"""

    external_tools = """
### Integraciones Externas

#### INE (Instituto Nacional de Estadistica)
- `ine_list_operations`: Listar operaciones estadisticas
- `ine_search_operations`: Buscar operaciones
- `ine_list_tables`: Listar tablas de una operacion
- `ine_get_data`: Obtener datos de una tabla

#### AEMET (Meteorologia)
- `aemet_list_stations`: Listar estaciones meteorologicas
- `aemet_list_municipalities`: Listar municipios
- `aemet_get_observations`: Obtener observaciones
- `aemet_get_forecast`: Obtener prediccion

#### BOE (Boletin Oficial del Estado)
- `boe_get_today`: Sumario de hoy
- `boe_get_summary`: Sumario de una fecha
- `boe_get_document`: Obtener documento por ID
- `boe_search`: Buscar documentos
"""

    utility_tools = """
### Herramientas de Utilidad

**export_results** - Exportar resultados a CSV/JSON
- `search_results`: JSON de busqueda previa
- `format`: 'csv' o 'json'
- `include_preview`: Incluir columnas de preview

**get_usage_stats** - Ver estadisticas de uso
- `include_searches`: Incluir consultas recientes

**clear_usage_stats** - Limpiar estadisticas de uso
"""

    # Build content based on category
    if tool_category == "search":
        tools_content = search_tools
    elif tool_category == "metadata":
        tools_content = metadata_tools
    elif tool_category == "external":
        tools_content = external_tools
    elif tool_category == "utilities":
        tools_content = utility_tools
    else:
        tools_content = search_tools + metadata_tools + external_tools + utility_tools

    return f"""# Guia de Herramientas MCP - datos.gob.es

## Descripcion General

Este servidor MCP proporciona acceso al catalogo de datos abiertos de Espana
(datos.gob.es) con 32 herramientas organizadas en categorias:

- **Busqueda**: Buscar y explorar datasets
- **Metadatos**: Listas de publicadores, temas, regiones
- **Externas**: INE, AEMET, BOE
- **Utilidades**: Exportar, metricas de uso

## Herramientas Disponibles
{tools_content}
{examples_section}
## Recursos MCP Disponibles

### Recursos Estaticos (acceso directo)
- `catalog://themes` - Todas las tematicas
- `catalog://publishers` - Todos los publicadores
- `catalog://provinces` - Provincias espanolas
- `catalog://autonomous-regions` - Comunidades Autonomas

### Resource Templates (dinamicos)
- `dataset://{{id}}` - Info de un dataset
- `theme://{{id}}` - Datasets de una tematica
- `publisher://{{id}}` - Datasets de un publicador
- `format://{{id}}` - Datasets en un formato
- `keyword://{{palabra}}` - Datasets con palabra clave

## Consejos de Uso

1. **Usa cache**: Los metadatos se cachean 24h. Segunda llamada es instantanea.
2. **Busqueda semantica**: Primera vez tarda 30-60s en construir indice.
3. **Paginacion paralela**: `fetch_all=true` es 5x mas rapido.
4. **Multi-tema**: Usa `themes=["economia", "empleo"]` para buscar en varios.
5. **Exportar**: Guarda resultados con `export_results` para analisis posterior.
6. **Metricas**: Usa `get_usage_stats` para ver que herramientas usas mas.
"""
