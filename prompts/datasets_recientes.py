"""Prompt para búsqueda de datasets actualizados recientemente."""

PROMPT_NAME = "datasets_recientes"
PROMPT_DESCRIPTION = """Búsqueda de datasets actualizados recientemente.

Este prompt te guía para encontrar los datasets que han sido
actualizados o publicados en los últimos días."""


def generate_prompt(
    dias: int = 30,
    tema: str | None = None,
    max_resultados: int = 15
) -> str:
    """
    Genera el prompt de búsqueda de datasets recientes.

    Args:
        dias: Número de días hacia atrás para buscar
        tema: Temática opcional para filtrar
        max_resultados: Número máximo de resultados
    """
    filtro_tema = f" sobre **{tema}**" if tema else ""
    instruccion_tema = f"""
### Paso 2: Filtrar por temática (opcional)
Si deseas filtrar por tema "{tema}", usa `get_datasets_by_theme` con los resultados.""" if tema else ""

    return f"""# Búsqueda de Datasets Actualizados Recientemente

## Objetivo
Encontrar datasets{filtro_tema} que hayan sido actualizados o publicados en los últimos **{dias} días** en datos.gob.es.

## Contexto
El portal datos.gob.es se actualiza constantemente con nuevos datasets y actualizaciones de datos existentes. Esta búsqueda te permite identificar las novedades más recientes.

## Instrucciones de Búsqueda

### Paso 1: Calcular el rango de fechas
- Fecha de inicio: hace {dias} días (formato: YYYY-MM-DDTHH:mmZ)
- Fecha de fin: hoy (formato: YYYY-MM-DDTHH:mmZ)

Usa la herramienta `get_datasets_by_date_range` con:
- begin_date: fecha de hace {dias} días
- end_date: fecha actual
- page_size: {max_resultados}
{instruccion_tema}

### Paso 3: Analizar los resultados
Para cada dataset encontrado, extrae:
- Título del dataset
- Descripción breve
- Organismo publicador
- Fecha exacta de modificación
- Temáticas asociadas
- Número de distribuciones disponibles

### Paso 4: Obtener detalles de los más relevantes
Para los datasets más interesantes, usa `dataset://{{dataset_id}}` para obtener:
- URLs de descarga directa
- Formatos disponibles
- Frecuencia de actualización
- Cobertura temporal de los datos

## Criterios de Priorización
Ordena los resultados por:
1. Fecha de modificación (más recientes primero)
2. Relevancia del publicador (organismos oficiales prioritarios)
3. Completitud de metadatos

## Resultado Esperado
Presenta un resumen de los {max_resultados} datasets más recientes{filtro_tema}, incluyendo:
- Estadísticas: total de datasets actualizados en el período
- Lista detallada con título, publicador, fecha y descripción
- Destacados: los 3 datasets más relevantes o interesantes
- Enlaces directos a los datos cuando estén disponibles"""
