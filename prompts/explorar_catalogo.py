"""Prompt para exploración guiada del catálogo de datos abiertos."""

PROMPT_NAME = "explorar_catalogo"
PROMPT_DESCRIPTION = """Exploración guiada del catálogo de datos abiertos.

Este prompt proporciona una guía completa para explorar el catálogo
de datos.gob.es y encontrar datasets relevantes para un interés específico."""


def generate_prompt(
    interes: str = "datos económicos de España"
) -> str:
    """
    Genera el prompt de exploración del catálogo.

    Args:
        interes: Descripción del área de interés para la búsqueda
    """
    return f"""# Exploración Guiada del Catálogo de Datos Abiertos de España

## Objetivo
Realizar una exploración exhaustiva del catálogo datos.gob.es para encontrar datasets relacionados con: **{interes}**

## Sobre datos.gob.es
El portal datos.gob.es es el catálogo nacional de datos abiertos de España, que reúne información de:
- Administración General del Estado
- Comunidades Autónomas
- Entidades Locales
- Universidades y organismos públicos

Contiene miles de datasets en múltiples formatos, actualizados regularmente.

## Estrategia de Exploración

### Fase 1: Reconocimiento del Catálogo

#### 1.1 Explorar temáticas disponibles
Consulta `catalog://themes` para identificar las categorías temáticas relacionadas con "{interes}".

#### 1.2 Identificar publicadores relevantes
Consulta `catalog://publishers` para encontrar organismos que puedan publicar datos sobre este tema.

#### 1.3 Revisar cobertura geográfica
Si el interés tiene componente territorial, consulta:
- `catalog://autonomous-regions` para datos autonómicos
- `catalog://provinces` para datos provinciales

### Fase 2: Búsqueda Sistemática

#### 2.1 Búsqueda por título
Usa `search_datasets_by_title` con palabras clave extraídas de "{interes}".
Prueba variaciones y sinónimos.

#### 2.2 Búsqueda por temática
Usa `get_datasets_by_theme` con las temáticas identificadas en Fase 1.

#### 2.3 Búsqueda por keywords
Usa `get_datasets_by_keyword` con términos específicos relacionados.

#### 2.4 Búsqueda cruzada por formato
Si necesitas datos en un formato específico (CSV, JSON, etc.), usa `get_datasets_by_format`.

### Fase 3: Análisis de Resultados

#### 3.1 Evaluación de datasets
Para cada dataset prometedor, obtén detalles completos con `dataset://{{dataset_id}}`:
- Calidad de los metadatos
- Frecuencia de actualización
- Cobertura temporal
- Formatos disponibles
- Licencia de uso

#### 3.2 Verificar distribuciones
Usa `get_distributions_by_dataset` para confirmar:
- URLs de descarga funcionan
- Formatos realmente disponibles
- Tamaño de los archivos

### Fase 4: Síntesis y Recomendaciones

#### 4.1 Categorización de hallazgos
Agrupa los datasets encontrados por:
- Fuente/Publicador
- Tipo de datos
- Actualización (tiempo real, mensual, anual)
- Formato preferido

#### 4.2 Recomendaciones priorizadas
Presenta los TOP 5 datasets más relevantes para "{interes}" con:
- Justificación de la relevancia
- Instrucciones de acceso
- Posibles usos de los datos
- Limitaciones conocidas

## Consejos para la Búsqueda
- Los títulos pueden estar en español e inglés
- Usa términos oficiales (ej: "presupuesto" vs "budget")
- Los IDs de tema no llevan tildes (ej: "economia" no "economía")
- La paginación empieza en 0
- Máximo 50 resultados por página

## Resultado Esperado
Un informe completo que incluya:
1. Resumen ejecutivo de hallazgos
2. Lista categorizada de datasets relevantes
3. Análisis de la calidad y actualidad de los datos
4. Recomendaciones de uso
5. Datasets relacionados para exploración futura"""
