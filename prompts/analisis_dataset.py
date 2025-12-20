"""Prompt para análisis detallado de un dataset específico."""

PROMPT_NAME = "analisis_dataset"
PROMPT_DESCRIPTION = """Análisis detallado de un dataset específico.

Este prompt te guía para realizar un análisis exhaustivo de un dataset
concreto del catálogo datos.gob.es, incluyendo metadatos, distribuciones,
calidad de datos y posibles usos."""


def generate_prompt(
    dataset_id: str = "",
    incluir_distribuciones: bool = True,
    evaluar_calidad: bool = True
) -> str:
    """
    Genera el prompt de análisis de dataset.

    Args:
        dataset_id: Identificador del dataset a analizar
        incluir_distribuciones: Si incluir análisis de distribuciones
        evaluar_calidad: Si incluir evaluación de calidad de datos
    """
    dataset_ref = f"**{dataset_id}**" if dataset_id else "el dataset especificado"

    seccion_distribuciones = """
### Fase 3: Análisis de Distribuciones

#### 3.1 Inventario de distribuciones
Usa `get_distributions_by_dataset` con el dataset_id para obtener todas las distribuciones.

Para cada distribución, documenta:
- **Formato**: CSV, JSON, XML, XLSX, RDF, etc.
- **URL de acceso**: Enlace directo de descarga
- **Tamaño**: Tamaño del archivo si está disponible
- **Media type**: Tipo MIME del recurso
- **Título**: Nombre descriptivo de la distribución

#### 3.2 Evaluación de accesibilidad
- Verificar que las URLs son accesibles
- Identificar si requiere autenticación
- Comprobar si hay APIs disponibles (SPARQL, REST)

#### 3.3 Comparativa de formatos
| Formato | Ventajas | Casos de uso recomendados |
|---------|----------|---------------------------|
| CSV | Universalmente compatible | Análisis en Excel, Python, R |
| JSON | Estructurado, APIs | Desarrollo web, aplicaciones |
| XML | Interoperabilidad | Integración con sistemas legacy |
| RDF | Datos enlazados | Web semántica, SPARQL |
""" if incluir_distribuciones else ""

    seccion_calidad = """
### Fase 4: Evaluación de Calidad

#### 4.1 Completitud de metadatos
Evalúa la presencia de:
- [ ] Título descriptivo y claro
- [ ] Descripción detallada del contenido
- [ ] Palabras clave relevantes
- [ ] Temáticas correctamente asignadas
- [ ] Información del publicador completa
- [ ] Licencia claramente especificada
- [ ] Frecuencia de actualización definida
- [ ] Cobertura temporal documentada
- [ ] Cobertura geográfica especificada

#### 4.2 Actualidad de los datos
- **Fecha de creación**: ¿Cuándo se publicó originalmente?
- **Última modificación**: ¿Cuándo se actualizó por última vez?
- **Frecuencia declarada vs real**: ¿Se cumple el calendario de actualización?
- **Vigencia**: ¿Los datos siguen siendo relevantes?

#### 4.3 Usabilidad
- **Documentación**: ¿Existe documentación adicional?
- **Diccionario de datos**: ¿Se describen los campos/columnas?
- **Ejemplos de uso**: ¿Hay ejemplos o tutoriales?
- **Contacto**: ¿Hay forma de contactar al publicador?

#### 4.4 Puntuación de calidad (0-100)
Asigna una puntuación basada en:
- Completitud de metadatos (25 puntos)
- Actualidad de datos (25 puntos)
- Accesibilidad de distribuciones (25 puntos)
- Documentación y usabilidad (25 puntos)
""" if evaluar_calidad else ""

    return f"""# Análisis Detallado de Dataset

## Objetivo
Realizar un análisis exhaustivo de {dataset_ref} del catálogo datos.gob.es, evaluando sus características, calidad y posibles aplicaciones.

## Información Requerida
- **Dataset ID**: {dataset_id if dataset_id else "(especificar el ID del dataset a analizar)"}

## Metodología de Análisis

### Fase 1: Obtención de Información Base

#### 1.1 Metadatos principales
Consulta el recurso `dataset://{dataset_id if dataset_id else "{dataset_id}"}` para obtener:
- Título completo
- Descripción detallada
- Organismo publicador
- Fecha de creación y última modificación
- Licencia de uso
- Frecuencia de actualización

#### 1.2 Clasificación temática
Identifica:
- Temáticas asignadas (theme)
- Palabras clave (keywords)
- Categorías relacionadas

#### 1.3 Cobertura
- **Temporal**: Período que cubren los datos
- **Geográfica**: Ámbito territorial (nacional, autonómico, local)
- **Sectorial**: Área de actividad o sector

### Fase 2: Contexto del Publicador

#### 2.1 Información del organismo
Usa `publisher://{{publisher_id}}` para conocer:
- Nombre completo del organismo
- Otros datasets publicados
- Patrón de publicación (frecuencia, temas)

#### 2.2 Datasets relacionados
Busca datasets relacionados usando:
- `get_datasets_by_theme` con las mismas temáticas
- `get_datasets_by_keyword` con las mismas palabras clave
- `search_datasets_by_title` con términos similares
{seccion_distribuciones}{seccion_calidad}
### Fase 5: Análisis de Aplicabilidad

#### 5.1 Casos de uso potenciales
Identifica posibles aplicaciones:
- **Análisis estadístico**: ¿Qué análisis se pueden realizar?
- **Visualización**: ¿Qué gráficos o mapas se pueden crear?
- **Integración**: ¿Con qué otros datos se puede combinar?
- **Aplicaciones**: ¿Qué apps o servicios se podrían desarrollar?

#### 5.2 Limitaciones identificadas
- Restricciones de la licencia
- Gaps en los datos (períodos sin información)
- Problemas de formato o estructura
- Necesidad de limpieza o transformación

#### 5.3 Datasets complementarios
Sugiere otros datasets que podrían enriquecer el análisis:
- Datos demográficos relacionados
- Series temporales comparables
- Datos geográficos complementarios

## Resultado del Análisis

### Ficha Técnica
Presenta un resumen estructurado:

```
DATASET: [Título]
ID: {dataset_id if dataset_id else "[ID]"}
PUBLICADOR: [Organismo]
LICENCIA: [Tipo de licencia]
ÚLTIMA ACTUALIZACIÓN: [Fecha]
FRECUENCIA: [Periodicidad]
FORMATOS: [Lista de formatos disponibles]
CALIDAD: [Puntuación]/100
```

### Recomendaciones
1. **Para usuarios principiantes**: Pasos para comenzar a usar los datos
2. **Para desarrolladores**: APIs y formatos recomendados
3. **Para analistas**: Herramientas y metodologías sugeridas
4. **Para investigadores**: Posibles líneas de investigación

### Conclusión
Proporciona una valoración final sobre:
- Utilidad general del dataset
- Público objetivo recomendado
- Prioridad de uso (alta/media/baja)
- Áreas de mejora sugeridas al publicador"""
