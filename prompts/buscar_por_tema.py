"""Prompt para búsqueda guiada de datasets por temática y formato."""

PROMPT_NAME = "buscar_datos_por_tema"
PROMPT_DESCRIPTION = """Búsqueda guiada de datasets por temática y formato.

Este prompt te guía para encontrar datasets de una temática específica
disponibles en un formato determinado."""


def generate_prompt(
    tema: str = "economia",
    formato: str = "csv",
    max_resultados: int = 10
) -> str:
    """
    Genera el prompt de búsqueda por tema y formato.

    Args:
        tema: Temática a buscar (economia, salud, educacion, etc.)
        formato: Formato deseado (csv, json, xml, xlsx, rdf)
        max_resultados: Número máximo de resultados a mostrar
    """
    return f"""# Búsqueda de Datos Abiertos por Temática

## Objetivo
Encontrar datasets del portal datos.gob.es sobre **{tema}** disponibles en formato **{formato}**.

## Instrucciones de Búsqueda

### Paso 1: Verificar la temática
Primero, consulta el recurso `catalog://themes` para verificar que "{tema}" es una temática válida.
Las temáticas principales son: economia, hacienda, educacion, salud, medio-ambiente, transporte, turismo, empleo, sector-publico, ciencia-tecnologia.

### Paso 2: Buscar datasets por tema
Usa la herramienta `get_datasets_by_theme` con:
- theme_id: "{tema}"
- page_size: {max_resultados}

### Paso 3: Filtrar por formato
De los resultados obtenidos, usa `get_datasets_by_format` con:
- format_id: "{formato}"

### Paso 4: Obtener detalles
Para cada dataset interesante, usa el recurso `dataset://{{dataset_id}}` para obtener información completa incluyendo URLs de descarga.

## Formatos Disponibles
- csv: Valores separados por comas (ideal para análisis)
- json: JavaScript Object Notation (ideal para APIs)
- xml: Extensible Markup Language
- xlsx: Microsoft Excel
- rdf: Resource Description Framework (datos enlazados)

## Criterios de Evaluación de Datasets
Al presentar los resultados, incluye:
1. Título y descripción del dataset
2. Organismo publicador
3. Fecha de última actualización
4. Formatos disponibles
5. URL de acceso a los datos

## Resultado Esperado
Presenta una lista de hasta {max_resultados} datasets sobre {tema} en formato {formato}, ordenados por fecha de modificación (más recientes primero)."""
