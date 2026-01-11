"""Prompt para busqueda guiada de estadisticas oficiales de Espana."""

PROMPT_NAME = "buscar_estadisticas"
PROMPT_DESCRIPTION = """Busqueda guiada de estadisticas oficiales espanolas.

Este prompt guia al asistente para buscar datos estadisticos tanto en
el INE (Instituto Nacional de Estadistica) como en datos.gob.es."""


def generate_prompt(
    tema: str = "empleo", incluir_ine: bool = True, incluir_datos_gob: bool = True
) -> str:
    """
    Genera el prompt de busqueda de estadisticas.

    Args:
        tema: Tema estadistico a buscar (empleo, poblacion, precios, turismo, etc.)
        incluir_ine: Si buscar en INE (fuente principal de estadisticas oficiales)
        incluir_datos_gob: Si buscar tambien en datos.gob.es
    """
    sources = []
    if incluir_ine:
        sources.append("INE")
    if incluir_datos_gob:
        sources.append("datos.gob.es")
    sources_str = " y ".join(sources)

    ine_section = ""
    if incluir_ine:
        ine_section = f"""
### Fuente 1: INE (Instituto Nacional de Estadistica) - FUENTE PRINCIPAL

El INE es la **fuente oficial y principal** de estadisticas en Espana. Siempre debe
consultarse primero para datos estadisticos oficiales.

**Paso 1.1: Buscar operaciones estadisticas**
Usa `ine_search(query="{tema}")` para encontrar operaciones relacionadas.

Ejemplos de operaciones INE relevantes:
- "empleo" -> EPA (Encuesta de Poblacion Activa), paro registrado
- "poblacion" -> Cifras de poblacion, censo, migraciones
- "precios" / "IPC" -> Indice de Precios al Consumo, inflacion
- "turismo" -> Estadisticas de turismo, viajeros, pernoctaciones
- "PIB" -> Contabilidad Nacional, crecimiento economico
- "vivienda" -> Precios de vivienda, hipotecas

**Paso 1.2: Explorar tablas disponibles**
Una vez identificada la operacion, usa `ine_search(operation_id="...")` para ver
las tablas de datos disponibles.

**Paso 1.3: Obtener datos**
Usa `ine_download(table_id, n_last=10)` para obtener los datos estadisticos reales.
"""

    datos_gob_section = ""
    if incluir_datos_gob:
        datos_gob_section = f"""
### Fuente 2: datos.gob.es - Catalogo de Datos Abiertos

El portal datos.gob.es contiene datasets de multiples organismos, incluyendo
datos que complementan las estadisticas del INE.

**Paso 2.1: Buscar datasets**
Usa `search(title="{tema}")` o `search(keyword="{tema}")`.

**Paso 2.2: Busqueda semantica (opcional)**
Para busquedas mas inteligentes, usa `search(query="{tema}")`.

**Paso 2.3: Obtener datos**
Usa `get(dataset_id)` para metadatos o `get(dataset_id, include_data=true)` para
descargar los datos directamente.
"""

    return f"""# Busqueda de Estadisticas Oficiales de Espana

## Objetivo
Encontrar datos estadisticos oficiales sobre **{tema}** consultando {sources_str}.

## IMPORTANTE: Orden de Consulta

Para estadisticas oficiales espanolas, SIEMPRE consulta en este orden:

1. **INE (Instituto Nacional de Estadistica)**: Fuente PRINCIPAL de estadisticas oficiales.
   Contiene datos de empleo (EPA), poblacion, precios (IPC), PIB, turismo, etc.

2. **datos.gob.es**: Complementa con datasets de otros organismos y formatos adicionales.
{ine_section}{datos_gob_section}
## Temas Estadisticos Comunes y Donde Buscar

| Tema | INE (Principal) | datos.gob.es (Complemento) |
|------|-----------------|---------------------------|
| Empleo/Paro | EPA, Paro registrado | Datasets de SEPE |
| Poblacion | Censo, Padron | Datos municipales |
| Precios/Inflacion | IPC, IPRI | Series historicas |
| Turismo | ETR, EGATUR | Datos regionales |
| PIB/Economia | Contabilidad Nacional | Indicadores economicos |
| Vivienda | Precios vivienda | Datos inmobiliarios |
| Educacion | Estadisticas educacion | Datos de universidades |
| Salud | Encuesta de salud | Datos sanitarios |

## Formato de Respuesta

Al presentar los resultados, incluye:

1. **Fuente**: Indicar si es del INE o de datos.gob.es
2. **Nombre**: Titulo de la operacion/dataset
3. **Datos disponibles**: Que informacion contiene
4. **Periodo temporal**: Desde cuando hasta cuando hay datos
5. **Frecuencia**: Mensual, trimestral, anual
6. **Valores recientes**: Si es posible, mostrar los ultimos datos

## Ejemplo de Flujo Completo para "{tema}"

```
1. ine_search(query="{tema}")
   -> Obtener IDs de operaciones relevantes

2. ine_search(operation_id="...")
   -> Ver tablas disponibles

3. ine_download(table_id="...", n_last=10)
   -> Obtener datos reales

4. search(title="{tema}")
   -> Buscar datasets complementarios en datos.gob.es
```
"""
