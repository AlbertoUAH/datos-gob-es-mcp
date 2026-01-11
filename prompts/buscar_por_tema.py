"""Prompt para busqueda guiada de datasets por tematica y formato."""

PROMPT_NAME = "buscar_datos_por_tema"
PROMPT_DESCRIPTION = """Busqueda guiada de datasets por tematica y formato.

Este prompt te guia para encontrar datasets de una tematica especifica
disponibles en un formato determinado."""


def generate_prompt(tema: str = "economia", formato: str = "csv", max_resultados: int = 10) -> str:
    """
    Genera el prompt de busqueda por tema y formato.

    Args:
        tema: Tematica a buscar (economia, salud, educacion, etc.)
        formato: Formato deseado (csv, json, xml, xlsx, rdf)
        max_resultados: Numero maximo de resultados a mostrar
    """
    return f"""# Busqueda de Datos Abiertos por Tematica

## Objetivo
Encontrar datasets del portal datos.gob.es sobre **{tema}** disponibles en formato **{formato}**.

## Instrucciones de Busqueda

### Paso 1: Verificar la tematica
Las tematicas validas son: economia, hacienda, educacion, salud, medio-ambiente,
transporte, turismo, empleo, sector-publico, ciencia-tecnologia, cultura-ocio,
urbanismo-infraestructuras, energia.

### Paso 2: Buscar datasets por tema y formato
Usa la herramienta `search` con:
- theme: "{tema}"
- format: "{formato}"
- max_results: {max_resultados}

Ejemplo: `search(theme="{tema}", format="{formato}", max_results={max_resultados})`

### Paso 3: Obtener detalles
Para cada dataset interesante, usa `get(dataset_id)` para obtener informacion
completa incluyendo URLs de descarga.

Para descargar datos directamente: `get(dataset_id, include_data=true)`

## Formatos Disponibles
- csv: Valores separados por comas (ideal para analisis)
- json: JavaScript Object Notation (ideal para APIs)
- xml: Extensible Markup Language
- xlsx: Microsoft Excel
- rdf: Resource Description Framework (datos enlazados)

## Criterios de Evaluacion de Datasets
Al presentar los resultados, incluye:
1. Titulo y descripcion del dataset
2. Organismo publicador
3. Fecha de ultima actualizacion
4. Formatos disponibles
5. URL de acceso a los datos

## Resultado Esperado
Presenta una lista de hasta {max_resultados} datasets sobre {tema} en formato {formato},
ordenados por fecha de modificacion (mas recientes primero)."""
