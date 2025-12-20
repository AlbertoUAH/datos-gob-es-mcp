A continuación tienes un **manual/catálogo detallado** de la **API de datos.gob.es (apidata)**, con **todas las llamadas publicadas** (Catálogo + NTI) y una guía práctica de **entradas, salidas, paginación, formatos** y **modelos de datos**.

---

## 1) Qué es “apidata” y cómo está montada

La API **apidata** es una **Linked Data API** (implementada con **ELDA**) que actúa como **gateway REST** sobre el **endpoint SPARQL** (Virtuoso) que contiene el grafo RDF del catálogo. ([Datos.gob.es][1])

Esto explica dos rasgos importantes:

1. Las respuestas suelen devolver **URIs** y entidades con propiedades típicas RDF/DCAT. ([Datos.gob.es][2])
2. Hay dos formas de acceso:

   * **REST (apidata)** para consultas predefinidas por rutas. ([Datos.gob.es][2])
   * **SPARQL** para consultas ad-hoc más flexibles. ([Datos.gob.es][3])

---

## 2) Convenciones comunes (aplican a todas las llamadas REST)

### 2.1 Base URL

Todas las rutas documentadas cuelgan de:

* `http://datos.gob.es/apidata/...` (en la documentación aparecen ejemplos en HTTP) ([Datos.gob.es][2])
* En la práctica, verás respuestas también sirviéndose bajo `https://datos.gob.es/apidata/...` (recomendable usar HTTPS cuando sea posible).

### 2.2 Formatos de salida (Content Negotiation)

La API soporta: **json, xml, rdf, ttl, csv**. ([Datos.gob.es][2])

Puedes seleccionar el formato de dos maneras: ([Datos.gob.es][2])

1. Cabecera **`Accept`** (p.ej. `Accept: application/rdf+xml`)
2. Extensión en la URL (p.ej. `.xml`, `.rdf`, `.turtle`, `.csv`, `.json`)

Tabla de MIME-types y extensiones (tal y como lo documenta datos.gob.es): ([Datos.gob.es][2])

* JSON → `application/json` / `.json`
* XML → `application/xml` / `.xml`
* RDF → `application/rdf+xml` / `.rdf`
* Turtle → `application/x-turtle` / `.turtle`
* CSV → `text/csv` / `.csv`

### 2.3 Parámetros de consulta (query string)

Parámetros estándar documentados: ([Datos.gob.es][2])

* **`_sort`**: ordenación por uno o varios campos.

  * Descendente: prefijo `-` (p.ej. `-issued`)
  * Múltiples campos: separados por coma
* **`_pageSize`**: tamaño de página (máximo **50**)
* **`_page`**: índice de página (empieza en **0**)

Ejemplos documentados: ([Datos.gob.es][2])

* `.../dataset.json?_sort=-issued,title`
* `.../dataset.xml?_pageSize=1&_page=2`

> Nota práctica: en respuestas reales aparece también `?_metadata=all` como “extendedMetadataVersion”. No está descrito en la tabla de parámetros de la versión accesible, pero sí se observa en payloads JSON. ([Datos.gob.es][4])

### 2.4 Estructura típica de respuesta (en JSON)

El “sobre” suele tener esta forma (alto nivel): ([Datos.gob.es][4])

* `format`: `"linked-data-api"`
* `version`: `"0.2"`
* `result`: objeto con:

  * `_about`, `definition`, `extendedMetadataVersion`
  * paginación (`first`, `next`, `itemsPerPage`, `page`, `startIndex`)
  * `items`: array con resultados (datasets, distribuciones o conceptos/URIs)

---

## 3) Catálogo completo de llamadas REST disponibles (datos.gob.es / apidata)

La documentación accesible lista **todas** estas llamadas, agrupadas en **Catálogo de datos** y **Norma Técnica de Interoperabilidad (NTI)**. ([Datos.gob.es][2])

### 3.1 Catálogo de datos (datasets, distributions, publishers, spatial, theme)

#### A) Conjuntos de datos (Datasets)

**1) Obtener todos los conjuntos de datos** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/dataset`
* **Entradas**:

  * Query params: `_sort`, `_pageSize` (≤50), `_page` ([Datos.gob.es][2])
  * Formato: `Accept` o extensión (`.json`, `.rdf`, etc.) ([Datos.gob.es][2])
* **Salida**:

  * Lista paginada (en JSON: `result.items` es un array de `dcat:Dataset` / DCAT-AP-ES)

**2) Obtener un dataset por su identificador (URI/id)** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/dataset/{id}`
* **Entrada (path param)**:

  * `{id}`: identificador textual del dataset en datos.gob.es (slug)
* **Salida**:

  * Lista con un único dataset (habitualmente), en el mismo sobre `linked-data-api`

**3) Buscar datasets por título (match parcial)** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/dataset/title/{title}`
* **Entrada (path param)**:

  * `{title}`: texto (puede ser parte del título)
* **Salida**:

  * Lista paginada de datasets que coinciden

**4) Datasets por publicador** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/dataset/publisher/{id}`
* **Entrada (path param)**:

  * `{id}`: identificador del publicador (p.ej. `A16003011` en el ejemplo)
* **Salida**:

  * Lista paginada de datasets

**5) Datasets por temática/categoría** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/dataset/theme/{id}`
* **Entrada (path param)**:

  * `{id}`: identificador de categoría (p.ej. `hacienda`)
* **Salida**:

  * Lista paginada de datasets

**6) Datasets por formato de distribución** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/dataset/format/{format}`
* **Entrada (path param)**:

  * `{format}`: formato (p.ej. `csv`)
* **Salida**:

  * Lista paginada de datasets con al menos una distribución en ese formato

**7) Datasets por etiqueta/keyword** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/dataset/keyword/{keyword}`
* **Entrada (path param)**:

  * `{keyword}`: término (p.ej. `gastos`)
* **Salida**:

  * Lista paginada de datasets

**8) Datasets por ámbito geográfico (2 segmentos)** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/dataset/spatial/{spatialWord1}/{spatialWord2}`
* **Entrada (path params)**:

  * `{spatialWord1}` y `{spatialWord2}` (en el ejemplo: `Autonomia/Pais-Vasco`)
* **Salida**:

  * Lista paginada de datasets

**9) Datasets modificados entre dos fechas** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/dataset/modified/begin/{beginDate}/end/{endDate}`
* **Entrada (path params)**:

  * `{beginDate}`, `{endDate}` con formato **`AAAA-MM-DDTHH:mmZ`** (ejemplo en doc) ([Datos.gob.es][2])
* **Salida**:

  * Lista paginada de datasets

---

#### B) Distribuciones (Distributions)

**10) Obtener todas las distribuciones** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/distribution`
* **Entradas**:

  * `_sort`, `_pageSize`, `_page` ([Datos.gob.es][2])
* **Salida**:

  * Lista paginada de `dcat:Distribution`

**11) Distribuciones por dataset** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/distribution/dataset/{id}`
* **Entrada (path param)**:

  * `{id}`: id del dataset
* **Salida**:

  * Lista paginada de distribuciones de ese dataset

**12) Distribuciones por formato** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/distribution/format/{format}`
* **Entrada (path param)**:

  * `{format}`: p.ej. `csv`
* **Salida**:

  * Lista paginada de distribuciones

---

#### C) Publicadores / Cobertura geográfica / Temáticas

**13) Obtener todos los publicadores** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/publisher`
* **Salida**:

  * Lista paginada de publicadores (normalmente como URIs y/o recursos con etiquetas)

**14) Obtener opciones de cobertura geográfica con datasets** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/spatial`
* **Salida**:

  * Lista paginada de ámbitos geográficos (URIs/conceptos)

**15) Obtener todas las categorías/temáticas con datasets** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/catalog/theme`
* **Salida**:

  * Lista paginada de temas. Ejemplo real muestra mezcla de URIs y objetos con `prefLabel`. ([Datos.gob.es][5])

---

### 3.2 Norma Técnica de Interoperabilidad (NTI)

Estas llamadas exponen la **taxonomía de sectores primarios** y la **identificación de cobertura geográfica** de los anexos IV y V de la NTI. ([Datos.gob.es][2])

#### D) Taxonomía de sectores primarios

**16) Obtener todos los sectores primarios** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/nti/public-sector`
* **Salida**:

  * Lista paginada de conceptos (habitualmente SKOS)

**17) Obtener un sector por id** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/nti/public-sector/{id}`
* **Entrada (path param)**:

  * `{id}`: p.ej. `comercio`
* **Salida**:

  * Recurso del concepto/sector (o lista con un ítem)

#### E) Identificación de cobertura geográfica

**18) Obtener todas las provincias** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/nti/territory/Province`

**19) Obtener una provincia por id** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/nti/territory/Province/{id}`
* **Entrada**:

  * `{id}`: p.ej. `Madrid`

**20) Obtener todas las Comunidades Autónomas** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/nti/territory/Autonomous-region`

**21) Obtener una Comunidad Autónoma por id** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/nti/territory/Autonomous-region/{id}`
* **Entrada**:

  * `{id}`: p.ej. `Comunidad-Madrid`

**22) Obtener el país (España)** ([Datos.gob.es][2])

* **Método**: GET
* **Ruta**: `/apidata/nti/territory/Country/España`

---

## 4) Manual de salidas: modelos de datos (qué campos esperar)

### 4.1 Modelo “Dataset” y “Distribution” (DCAT-AP-ES)

Los recursos del catálogo siguen el perfil **DCAT-AP-ES** (Catálogo, Dataset, Distribución, etc.), basado en DCAT. ([Datos.gob.es][6])

**En la práctica**, un dataset en JSON (apidata) suele incluir campos equivalentes a:

* Identidad: `_about` (URI del recurso)
* Textos multilingües: `title`, `description` con estructuras tipo `{ "_value": "...", "_lang": "es" }` ([Datos.gob.es][7])
* Relación con distribuciones: `distribution` (array de `dcat:Distribution`) ([Datos.gob.es][7])
* Metadatos temporales: `issued`, `modified`
* Clasificación: `theme`, `keyword`
* Cobertura: `spatial`
* Periodicidad: `accrualPeriodicity` ([Datos.gob.es][4])

**Una distribución** suele incluir:

* `_about` (URI)
* `accessURL` (URL de acceso/descarga)
* `format` (habitualmente como `dct:IMT` / MIME)
* `identifier`
* `title` (posiblemente multilingüe)
* `type` = `http://www.w3.org/ns/dcat#Distribution` ([Datos.gob.es][7])

> Si necesitas un “contrato” de campos obligatorio/recomendado/opcional, la referencia normativa útil es el propio **DCAT-AP-ES**, que detalla clases y propiedades. ([Datos.gob.es][6])

### 4.2 Modelo “Theme” (y, por analogía, listas de conceptos)

En endpoints como `/catalog/theme` se observa que `result.items` puede mezclar:

* URIs como string
* Objetos con `_about` y `prefLabel` ([Datos.gob.es][5])

Esto es típico de listados de vocabularios/conceptos (SKOS).

---

## 5) Manual del endpoint SPARQL (alternativa avanzada)

Además de apidata (REST), datos.gob.es publica un **endpoint SPARQL**:

* **URL**: `http://datos.gob.es/virtuoso/sparql` ([Datos.gob.es][3])
* **Método**: GET (y típicamente POST, aunque en la página se ilustra por URL)

**Parámetros típicos** (según la página de ejemplos):

* `query`: la consulta SPARQL (URL-encoded)
* `format`: formato de respuesta (`application/rdf+xml`, `application/sparql-results+xml`, `application/sparql-results+json`, `text/csv`, `text/plain`, `text/turtle`, etc.) ([Datos.gob.es][3])

Ejemplo (plantilla) en cURL:

```bash
curl -G 'http://datos.gob.es/virtuoso/sparql' \
  --data-urlencode 'query=SELECT * WHERE { ?s ?p ?o } LIMIT 10' \
  --data-urlencode 'format=application/sparql-results+json'
```

---

## 6) Ejemplos prácticos (plantillas de consumo)

### 6.1 Listar datasets paginados y ordenados

(Orden por `issued` descendente y luego por `title`) ([Datos.gob.es][2])

```bash
curl 'http://datos.gob.es/apidata/catalog/dataset.json?_sort=-issued,title&_pageSize=50&_page=0'
```

### 6.2 Buscar datasets por palabra en el título

```bash
curl 'http://datos.gob.es/apidata/catalog/dataset/title/empleo.json?_pageSize=50&_page=0'
```

(“empleo” aparece como ejemplo de `{title}` en la documentación.) ([Datos.gob.es][2])

### 6.3 Datasets modificados en una ventana temporal

(Con el formato `AAAA-MM-DDTHH:mmZ`) ([Datos.gob.es][2])

```bash
curl 'http://datos.gob.es/apidata/catalog/dataset/modified/begin/2016-04-18T00:00Z/end/2016-06-30T00:00Z.json?_pageSize=50&_page=0'
```

---

## 7) Recomendación de “manual operativo” para tu equipo (cómo usarla bien)

1. **Descubrimiento**: usa `/catalog/theme`, `/catalog/publisher`, `/catalog/spatial` para “enumerar filtros” disponibles. ([Datos.gob.es][2])
2. **Consulta principal**: `/catalog/dataset` + filtros por ruta + `_sort/_pageSize/_page`. ([Datos.gob.es][2])
3. **Extracción de distribuciones**: o bien desde el propio dataset (`distribution`), o con `/catalog/distribution/dataset/{id}`. ([Datos.gob.es][2])
4. **Casos complejos** (agregaciones, joins, facetas no cubiertas por rutas): SPARQL. ([Datos.gob.es][3])

---

### Limitación relevante (transparente)

En la documentación accesible de apidata se publican **rutas, parámetros y formatos**, pero no un **OpenAPI/Swagger** con esquemas exhaustivos “tipo JSON Schema”. Por eso, para un contrato de campos “completo” conviene basarse en:

* el **perfil DCAT-AP-ES** (modelo de metadatos), ([Datos.gob.es][6])
* y la **observación de payloads reales** (que ya muestran `linked-data-api` + recursos DCAT/SKOS). ([Datos.gob.es][4])

---

Si me dices **para qué lo vais a usar** (p.ej., “cosecha diaria a un data lake”, “buscador interno por temática y provincia”, “catálogo empresarial de oportunidades”), puedo convertir este manual en una **especificación de integración**: contratos de datos esperados, estrategia de paginación, normalización de multilingüe, y un set de consultas SPARQL ya preparadas (incluyendo equivalencias REST ↔ SPARQL).

[1]: https://datos.gob.es/es/tecnologia "Tecnología | datos.gob.es"
[2]: https://datos.gob.es/es/accessible-apidata "API - Datos.gob.es"
[3]: https://datos.gob.es/es/ejemplos-sparql "Ejemplos SPARQL | datos.gob.es"
[4]: https://datos.gob.es/apidata/catalog/dataset/ea0010587-principales-variables-economicas-por-agrupaciones-de-actividad-anual-nacional-estadistica-estructural-de-empresas-sector-industrial-identificador-api-25321.json "datos.gob.es"
[5]: https://datos.gob.es/apidata/catalog/theme "datos.gob.es"
[6]: https://datosgobes.github.io/DCAT-AP-ES/?utm_source=chatgpt.com "DCAT-AP-ES - datosgobes.github.io"
[7]: https://datos.gob.es/apidata/catalog/dataset/a09002970-proyectos-finanzados-por-el-fondo-europeo-de-desarrollo-regional-feder-datos-de-evaluacion.json "datos.gob.es"
