# Guion de Demo - OpenData Hub MCP Server

## Introduccion (30 segundos)

**Narrador:**
> "Bienvenidos a la demo del Hub de OpenData Espanol - un servidor MCP que unifica el acceso a las principales fuentes de datos abiertos de Espana en una sola interfaz. Conecta con datos.gob.es, INE, AEMET y BOE."

---

## PARTE 1: Busqueda de Datasets (3-4 minutos)

### 1.1 Busqueda Simple por Titulo

**Pregunta al asistente:**
> "Busca datasets sobre presupuestos"

**Respuesta esperada (datos reales verificados del MCP):**
```
Encontrados 50 datasets en la primera pagina:

1. "Programa de inversiones de los presupuestos generales"
   URI: datos.gob.es/catalogo/a15002917-programa-de-inversiones-de-los-presupuestos-generales
   Publisher: Navarra | Formatos: HTML, XML, Excel, JSON, CSV, TSV

2. "Presupuestos. Presupuesto General 2025"
   URI: datos.gob.es/catalogo/l01280796-presupuestos-presupuesto-general-2025
   Publisher: Ayuntamiento de Madrid | Formatos: Excel, CSV

3. "Presupuestos de ingresos y gastos de las Entidades Locales segun clasificacion economica. Canarias. 2012"
   URI: datos.gob.es/catalogo/a05003423-presupuestos-de-ingresos-y-gastos...

4. "Presupuestos de la Comunidad Autonoma de Castilla-La Mancha desde el ano 2011 hasta 2021"
   URI: datos.gob.es/catalogo/a08002880-presupuestos-de-la-comunidad-autonoma...

5. "Presupuestos municipales iniciales consolidados. Gastos totales por habitante. Municipios"
   URI: datos.gob.es/catalogo/a13002908-presupuestos-municipales-iniciales...
```

---

### 1.2 Busqueda por Tematica

**Pregunta al asistente:**
> "Muestra datasets de economia"

**Respuesta esperada (datos reales):**
```
Datasets de la tematica Economia:

1. "Gross value added. CNEAG (Identificador API: 67197)"
2. "Final treatment of waste by type of waste, hazard and type of treatment"
3. "National indices by subgroup. IPCA (Identificador API: 67257)"
4. "P.I.B. a precios de mercado y valor anadido bruto a precios basicos..."
```

---

### 1.3 Busqueda Geografica (Espacial)

**Pregunta al asistente:**
> "Busca datos de empleo en la Comunidad de Madrid"

**Herramienta utilizada:** `search_datasets(title="empleo", spatial_type="Autonomia", spatial_value="Madrid")`

**Nota para demo:** El sistema busca automaticamente en metadatos espaciales, titulos y URIs.

---

### 1.4 Busqueda sobre COVID-19

**Pregunta al asistente:**
> "Encuentra datasets sobre COVID"

**Respuesta esperada (datos reales):**
```
Encontrados 10+ datasets sobre COVID-19:

1. "Dosis administradas de vacunas COVID-19. Campana 25/26"
   URI: datos.gob.es/catalogo/a07002862-dosis-administradas-de-vacunas-covid-19...

2. "Dosis de vacunas recibidas COVID-19 por marca y provincia. Campana 2025-2026"

3. "Personas vacunadas COVID-19 por fecha y provincia. Campana 25/26"

4. "Personas vacunadas de COVID-19 por criterio de vacunacion, provincia y fecha"
```

---

## PARTE 2: Metadatos y Catalogos (2-3 minutos)

### 2.1 Listar Tematicas Disponibles

**Pregunta al asistente:**
> "Cuales son las tematicas disponibles en el catalogo?"

**Herramienta:** `list_themes()`

**Respuesta esperada (37 temas disponibles):**
```
Tematicas disponibles en el catalogo (37 total):

Temas europeos (EU Data Theme):
- EDUC: Educacion
- TRAN: Transporte
- ENVI: Medio ambiente
- ECON: Economia
- GOVE: Gobierno y sector publico

Sectores NTI espanoles:
- sector-publico: Sector publico
- medio-ambiente: Medio ambiente
- sociedad-bienestar: Sociedad y bienestar
- empleo: Empleo
- economia: Economia
- industria: Industria
- demografia: Demografia
- comercio: Comercio
- turismo: Turismo
- salud: Salud
- ciencia-tecnologia: Ciencia y tecnologia
- cultura-ocio: Cultura y ocio
- educacion: Educacion
...
```

---

### 2.2 Sectores Publicos (NTI)

**Pregunta al asistente:**
> "Muestrame los sectores publicos definidos por la NTI"

**Herramienta:** `list_public_sectors()`

**Respuesta esperada (22 sectores NTI verificados):**
```
Sectores publicos definidos por la NTI (22 total):

1. sector-publico: Sector publico
2. medio-ambiente: Medio ambiente
3. sociedad-bienestar: Sociedad y bienestar
4. empleo: Empleo
5. economia: Economia
6. industria: Industria
7. demografia: Demografia
8. comercio: Comercio
9. turismo: Turismo
10. salud: Salud
11. ciencia-tecnologia: Ciencia y tecnologia
12. cultura-ocio: Cultura y ocio
13. deporte: Deporte
14. educacion: Educacion
15. energia: Energia
16. hacienda: Hacienda
17. justicia: Justicia
18. legislacion: Legislacion
19. seguridad: Seguridad
20. transporte: Transporte
21. urbanismo: Urbanismo e infraestructuras
22. vivienda: Vivienda
```

---

### 2.3 Comunidades Autonomas

**Pregunta al asistente:**
> "Lista las Comunidades Autonomas disponibles"

**Herramienta:** `list_autonomous_regions()`

**Respuesta esperada:**
```
17 Comunidades Autonomas + 2 ciudades autonomas:
- Andalucia
- Aragon
- Asturias
- Islas Baleares
- Canarias
- Cantabria
- Castilla-La Mancha
- Castilla y Leon
- Cataluna
- Comunidad Valenciana
- Extremadura
- Galicia
- Madrid
- Murcia
- Navarra
- Pais Vasco
- La Rioja
- Ceuta
- Melilla
```

---

## PARTE 3: Integracion INE (2-3 minutos)

### 3.1 Operaciones Estadisticas

**Pregunta al asistente:**
> "Que operaciones estadisticas tiene el INE?"

**Herramienta:** `ine_list_operations()`

**Respuesta esperada (datos reales):**
```
El INE tiene 110 operaciones estadisticas disponibles:

1. [30147] Estadistica de Efectos de Comercio Impagados
2. [30211] Indice de Coste Laboral Armonizado
3. [30168] Estadistica de Transmision de Derechos de la Propiedad
4. [30256] Indicadores Urbanos
5. [30219] Estadistica del Procedimiento Concursal
6. [30182] Indices de Precios del Sector Servicios
7. [30457] Indice de Precios de la Vivienda (IPV)
8. Distribucion de Nombres
...
```

---

### 3.2 Datos de Empleo (EPA)

**Pregunta al asistente:**
> "Dame datos de la Encuesta de Poblacion Activa"

**Herramienta:** `ine_get_data(operation_id="30245")`

**Respuesta esperada:**
```
Datos de la EPA (Encuesta de Poblacion Activa):
- Tasa de empleo de personas entre 20 y 64 anos
- Desglose por sexo y grupo de edad
- Serie temporal anual
- Ambito: Nacional
```

---

## PARTE 4: Integracion BOE (2-3 minutos)

### 4.1 Sumario del BOE de Hoy

**Pregunta al asistente:**
> "Muestrame el BOE de hoy"

**Herramienta:** `boe_get_today()`

**Nota importante:** El BOE no se publica en fines de semana ni festivos. Si no hay BOE disponible, la herramienta devolvera el mas reciente.

**Respuesta esperada (ejemplo BOE 30 Dic 2024 - datos reales):**
```
BOE - Sumario del 30 de Diciembre de 2024
=========================================
Numero: 314
Identificador: BOE-S-2024-314
PDF: https://www.boe.es/boe/dias/2024/12/30/pdfs/BOE-S-2024-314.pdf

Secciones:
- [1] I. Disposiciones generales
  -> Ministerio de Politica Territorial y Memoria Democratica
  -> Real Decreto 1309/2024: Traspaso de funciones a la Comunidad Autonoma del Pais Vasco

- [2A] II. Autoridades y personal - A. Nombramientos
- [2B] II. Autoridades y personal - B. Oposiciones y concursos
- [3] III. Otras disposiciones
- [4] IV. Administracion de Justicia
- [5] V. Anuncios
```

---

### 4.2 Buscar en BOE

**Pregunta al asistente:**
> "Busca disposiciones sobre subvenciones en el BOE"

**Herramienta:** `boe_search(query="subvenciones")`

---

## PARTE 5: Descarga y Preview de Datos (2-3 minutos)

### 5.1 Preview de Datos CSV

**Pregunta al asistente:**
> "Muestrame una preview de los datos de poblacion de Canarias"

**Herramienta:** `search_datasets(title="poblacion", include_preview=True, preview_rows=10)`

**Respuesta esperada (datos reales):**
```
Dataset: Poblacion de 16 y mas anos cuyo destino en vacaciones fue otra isla

Preview de datos (CSV):
+------------+---------------+--------------------------------+
| indicadores| islas_destino | poblacion_16_mas_anios_viaja   |
+------------+---------------+--------------------------------+
| Absoluto   | Total         | 206,274                        |
| Absoluto   | Lanzarote     | 22,158                         |
| Absoluto   | Fuerteventura | 54,322                         |
| Absoluto   | Gran Canaria  | 35,799                         |
| Absoluto   | Tenerife      | 28,910                         |
| Absoluto   | La Gomera     | 35,351                         |
| Absoluto   | La Palma      | 20,409                         |
| Absoluto   | El Hierro     | 9,325                          |
| Porcentaje | Total         | 100.00                         |
+------------+---------------+--------------------------------+
```

---

### 5.2 Descarga Completa

**Pregunta al asistente:**
> "Descarga el dataset completo de presupuestos de Madrid"

**Herramienta:** `download_data(dataset_id="l01280796-presupuestos...", max_mb=10)`

**Nota:** Permite descargar hasta 50MB de datos en formato CSV/JSON.

---

## PARTE 6: Busqueda Semantica con IA (2-3 minutos)

### 6.1 Busqueda por Significado

**Pregunta al asistente:**
> "Encuentra datos sobre desempleo juvenil en ciudades costeras"

**Herramienta:** `search_datasets(semantic_query="desempleo juvenil en ciudades costeras")`

**Nota para demo:**
- Primera busqueda puede tardar 30-60 segundos (construccion del indice)
- Busquedas posteriores son instantaneas (<1 segundo)
- Usa embeddings de IA para entender el significado

---

### 6.2 Datasets Relacionados

**Pregunta al asistente:**
> "Encuentra datasets similares al de tasa de empleo"

**Herramienta:** `get_related_datasets(dataset_id="ea0010587-tasa-de-empleo...", top_k=5)`

**Respuesta esperada:**
```
Datasets relacionados (por similitud semantica):

1. Tasa de paro por sexo y grupo de edad (score: 0.89)
2. Poblacion activa por nivel de estudios (score: 0.85)
3. Contratos registrados por tipo (score: 0.78)
4. Afiliados a la Seguridad Social (score: 0.75)
5. Indices de coste laboral (score: 0.72)
```

---

## PARTE 7: Exportacion y Metricas (1-2 minutos)

### 7.1 Exportar Resultados

**Pregunta al asistente:**
> "Exporta los resultados de busqueda a CSV"

**Herramienta:** `export_results(search_results="...", format="csv")`

**Respuesta esperada:**
```
Exportacion completada:
- Formato: CSV
- Filas exportadas: 25
- Columnas: uri, title, description, publisher, theme, modified, keyword, format
- Archivo: datasets_export.csv
```

---

### 7.2 Estadisticas de Uso

**Pregunta al asistente:**
> "Muestrame las estadisticas de uso de esta sesion"

**Herramienta:** `get_usage_stats(include_searches=True)`

**Respuesta esperada:**
```
Estadisticas de uso:
====================
Herramientas mas usadas:
1. search_datasets: 8 llamadas
2. list_themes: 3 llamadas
3. get_dataset: 2 llamadas
4. ine_list_operations: 1 llamada
5. boe_get_summary: 1 llamada

Datasets mas accedidos:
1. tasa-de-empleo-personas-20-64: 3 accesos
2. presupuestos-2025: 2 accesos

Busquedas recientes:
- "presupuestos" (filtro titulo)
- "empleo" + Madrid (filtro espacial)
- "desempleo juvenil" (semantica)
```

---

## PARTE 8: Webhooks y Notificaciones (1-2 minutos)

### 8.1 Registrar Vigilancia

**Pregunta al asistente:**
> "Quiero recibir notificaciones cuando se actualice el dataset de COVID"

**Herramienta:** `webhook_register(dataset_id="a07002862-dosis-administradas...", url="https://mi-servidor.com/webhook")`

**Respuesta esperada:**
```
Webhook registrado correctamente:
- Dataset: Dosis administradas de vacunas COVID-19
- URL de notificacion: https://mi-servidor.com/webhook
- Estado: Activo
- Se notificara cuando cambie la fecha de modificacion
```

---

### 8.2 Verificar Cambios

**Pregunta al asistente:**
> "Comprueba si hay cambios en los datasets que estoy vigilando"

**Herramienta:** `check_dataset_changes()`

---

## Resumen Final (30 segundos)

**Narrador:**
> "Hemos visto las principales capacidades del Hub de OpenData Espanol:
>
> - **32 herramientas** para acceder a multiples APIs publicas
> - **Busqueda inteligente** con filtros, semantica e hibrida
> - **Integraciones** con INE, AEMET y BOE
> - **Preview de datos** sin necesidad de descargar
> - **Exportacion** a CSV y JSON
> - **Metricas de uso** para analizar patrones
> - **Webhooks** para notificaciones de cambios
>
> Todo accesible desde cualquier cliente MCP como Claude Desktop."

---

## Comandos Rapidos de Referencia

| Accion | Herramienta | Ejemplo |
|--------|-------------|---------|
| Buscar datasets | `search_datasets` | `title="empleo"` |
| Busqueda semantica | `search_datasets` | `semantic_query="paro juvenil"` |
| Filtrar por tema | `search_datasets` | `theme="economia"` |
| Filtrar por region | `search_datasets` | `spatial_type="Autonomia", spatial_value="Madrid"` |
| Ver dataset | `get_dataset` | `dataset_id="..."` |
| Descargar datos | `download_data` | `dataset_id="...", max_mb=10` |
| Datos INE | `ine_get_data` | `operation_id="30245"` |
| BOE del dia | `boe_get_today` | - |
| Exportar CSV | `export_results` | `format="csv"` |
| Ver metricas | `get_usage_stats` | `include_searches=True` |

---

*Guion generado y verificado con datos reales del MCP en https://datos-gob-es.fastmcp.app/mcp*
*Todas las respuestas han sido probadas contra el servidor MCP real*
*Fecha: Enero 2025*
