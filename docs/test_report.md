# 🧪 MCP Tools Test Report

> **Generated:** 2026-01-11 15:02:25
> **Status:** ⚠️ 3 Failed
> **Total Tests:** 28

## 📊 Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Tests Pasados | 25 ✅ |
| Tests Fallidos | 3 ❌ |
| Tiempo Total | 10819 ms |
| Tiempo Promedio | 386 ms |

### Por Categoría

| Categoría | Pasados | Fallidos | Tiempo (ms) |
|-----------|---------|----------|-------------|
| ✅ datos.gob.es | 7 | 0 | 623 |
| ✅ INE | 8 | 0 | 5505 |
| ⚠️ AEMET | 5 | 2 | 2684 |
| ⚠️ BOE | 5 | 1 | 2007 |

## 🔧 Resultados Detallados

### 📦 datos.gob.es

#### `search`

**✅ Búsqueda por título (poblacion)** (151ms)

- **Parámetros:** `{"title": "poblacion", "max_results": 5}`
- **Resultado:** Keys: ['total', 'datasets']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'total': 0, 'datasets': [{'id': 'a05003423-poblacion-de-16-y-mas-anos-cuyo-destino-en-vacaciones-fue-otra-isla-del-archipielago-segun-islas-de-destino-2013', 'title': 'Poblacion de 16 y más años cuyo destino en vacaciones fue otra isla del archipiélago según islas de destino de Canarias. 2013'}, {'id': 'a05003423-poblacion-de-16-y-mas-anos-cuyo-destino-en-vacaciones-fue-otra-isla-del-archipielago-segun-islas-de-origen-y-de-destino-canarias-2007', 'title': 'Poblacion de 16 y más años cuyo destino en vacaciones fue otra isla del archipiélago según islas de origen y de destino. Canarias. 2007'}, {'id': 'a05003423-poblacion-de-16-y-mas-anos-cuyo-destino-en-vacaciones-fue-otra-isla-del-archipielago-segun-medios-de-transporte-interinsular-canarias-2013', 'title': 'Poblacion de 16 y más años cuyo destino en vacaciones fue otra isla del archipiélago según medios de transporte interinsular. Canarias. 2013'}, {'id': 'a05003423-poblacion-de-16-y-mas-anos-cuyo-destino-en-vacaciones-fue-otra-isla-del-archipielago-segun-medios-de-transporte-utilizados-durante-las-vacaciones-canarias-2013', 'title': 'Poblacion de 16 y más años cuyo destino en vacaciones fue otra isla del archipiélago según medios de transporte utilizados durante las vacaciones. Canarias. 2013'}, {'id': 'a05003423-poblacion-de-16-y-mas-anos-cuyo-destino-en-vacaciones-fue-otra-isla-del-archipielago-segun-tipos-de-alojamiento-canarias-2013', 'title': 'Poblacion de 16 y más años cuyo destino en vacaciones fue otra isla del archipiélago según tipos de alojamiento. Canarias. 2013'}]}
```
</details>

**✅ Búsqueda por tema (economía)** (96ms)

- **Parámetros:** `{"theme": "economia", "max_results": 5}`
- **Resultado:** Keys: ['total', 'datasets']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'total': 0, 'datasets': [{'id': 'ea0010587-valor-anadido-bruto-cneag-identificador-api-67197', 'title': 'Gross value added. CNEAG (Identificador API: 67197)'}, {'id': 'ea0010587-tratamiento-final-de-residuos-por-tipos-de-residuos-peligrosidad-y-tipo-de-tratamiento-identificador-api-67251', 'title': 'Final treatment of waste by type of waste, hazard and type of treatment. (Identificador API: 67251)'}, {'id': 'ea0010587-indices-nacionales-de-subgrupos-ipca-identificador-api-67257', 'title': 'National indices by subgroup. IPCA (Identificador API: 67257)'}, {'id': 'ea0010587-p-i-b-a-precios-de-mercado-y-valor-anadido-bruto-a-precios-basicos-por-ramas-de-actividad-por-comunidades-y-ciudades-autonomas-magnitud-y-periodo-identificador-api-67297', 'title': 'P.I.B. a precios de mercado y valor añadido bruto a precios básicos por ramas de actividad por comunidades y ciudades autónomas, magnitud y periodo. (Identificador API: 67297)'}, {'id': 'ea0010587-p-i-b-a-precios-de-mercado-y-valor-anadido-bruto-a-precios-basicos-por-ramas-de-actividad-precios-corrientes-por-comunidades-y-ciudades-autonomas-magnitud-y-periodo-identificador-api-67295', 'title': 'P.I.B. a precios de mercado y valor añadido bruto a precios básicos por ramas de actividad: Precios corrientes por comunidades y ciudades autónomas, magnitud y periodo. (Identificador API: 67295)'}]}
```
</details>

**✅ Búsqueda por formato CSV** (86ms)

- **Parámetros:** `{"format": "csv", "max_results": 5}`
- **Resultado:** Keys: ['total', 'datasets']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'total': 0, 'datasets': [{'id': 'l01300243-contratos-menores-2020', 'title': 'Contratos menores 2020'}, {'id': 'a10002983-agenda-cultural-del-institut-valencia-de-cultura-ivc', 'title': 'Agenda cultural del Institut Valencià de Cultura (IVC)'}, {'id': 'a10002983-microrreservas-de-flora-de-la-comunitat-valenciana', 'title': 'Microrreservas de flora de la Comunitat Valenciana'}, {'id': 'a10002983-periodo-medio-de-pago-a-beneficiarios-de-subvenciones-2015-2023', 'title': 'Periodo medio de pago a beneficiarios de subvenciones desde 2024'}, {'id': 'a10002983-trafico-diario-de-las-carreteras-que-gestiona-la-conselleria-competente-2025', 'title': 'Tráfico diario de las carreteras que gestiona la Conselleria competente 2025'}]}
```
</details>

**✅ Búsqueda por keyword (estadistica)** (126ms)

- **Parámetros:** `{"keyword": "estadistica", "max_results": 5}`
- **Resultado:** Keys: ['total', 'datasets']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'total': 0, 'datasets': [{'id': 'e05024301-registro-de-aparatos-electricos-y-electronicos1', 'title': 'Registration of Electric Appliances and Electronics'}, {'id': 'a16003011-estadistica-de-edificacion-y-vivienda-de-c-a-de-euskadi-3-trimestre-20141', 'title': 'EAEko Eraikuntzari eta Etxebizitzari Buruzko Estatistika, 2014ko 3. Hiruhilekoa.'}, {'id': 'a16003011-tablas-estadisticas-de-edificacion-y-vivienda-de-2013-edyvi1', 'title': '2013ko Eraikuntzari eta Etxebizitzari buruzko taula estatistikoak (EEEBE)'}, {'id': 'a16003011-tablas-estadisticas-de-edificacion-y-vivienda-de-2014-edyvi1', 'title': '2014ko Eraikuntzari eta Etxebizitzari buruzko taula estatistikoak (EEEBE)'}, {'id': 'a16003011-tablas-estadisticas-de-edificacion-y-vivienda-de-2015-edyvi1', 'title': '2015eko Eraikuntzari eta Etxebizitzari buruzko taula estatistikoak (EEEBE)'}]}
```
</details>


#### `get`

**✅ Obtener metadatos de dataset INE** (62ms)

- **Parámetros:** `{"dataset_id": "ea0010587-valor-anadido-bruto-cneag-identificador-api-67197"}`
- **Resultado:** Keys: ['id', 'title', 'description', 'publisher', 'distributions']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'id': 'ea0010587-valor-anadido-bruto-cneag-identificador-api-67197', 'title': 'Gross value added. CNEAG (Identificador API: 67197)', 'description': 'Tabla de INEbase\nValor Añadido Bruto. Anual. Nacional. Contabilidad nacional anual de España: agregados por rama de actividad', 'publisher': 'http://datos.gob.es/recurso/sector-publico/org/Organismo/EA0010587', 'distributions': 6}
```
</details>

**✅ Obtener dataset medio ambiente** (52ms)

- **Parámetros:** `{"dataset_id": "e05068001-mapas-estrategicos-de-ruido"}`
- **Resultado:** Keys: ['id', 'title', 'description', 'publisher', 'distributions']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'id': 'e05068001-mapas-estrategicos-de-ruido', 'title': 'Mapas estratégicos de Ruido', 'description': 'Mapas estratégicos de ruido (MER) de la tercera fase de implementación de la Directiva 2002/49/CE del Parlamento Europeo y del Consejo, de 25 de junio de 2002, sobre evaluación y gestión del ruido amb', 'publisher': 'http://datos.gob.es/recurso/sector-publico/org/Organismo/E05068001', 'distributions': 1}
```
</details>

**✅ Dataset inexistente (error esperado)** (50ms)

- **Parámetros:** `{"dataset_id": "dataset-que-no-existe-12345"}`
- **Resultado:** Expected error response: Dataset not found

<details>
<summary>📄 Respuesta completa</summary>

```json
{'error': 'Dataset not found', 'total': 0}
```
</details>


### 📊 INE

#### `ine_search`

**✅ Buscar operaciones (empleo)** (1329ms)

- **Parámetros:** `{"query": "empleo"}`
- **Resultado:** Keys: ['query', 'total_operations', 'operations']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'query': 'empleo', 'total_operations': 5, 'operations': [{'Id': 6, 'Cod_IOE': '30211', 'Nombre': 'Índice de Coste Laboral Armonizado', 'Codigo': 'ICLA', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736053992&idp=1254735976596'}, {'Id': 137, 'Cod_IOE': '30185', 'Nombre': 'Índice de Precios del Trabajo', 'Codigo': 'IPT', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736177027&idp=1254735976596'}, {'Id': 139, 'Cod_IOE': '30188', 'Nombre': 'Encuesta Anual de Coste Laboral', 'Codigo': 'EACL', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736060920&idp=1254735976596'}, {'Id': 234, 'Cod_IOE': '30209', 'Nombre': 'Estadística de Movilidad Laboral y Geográfica', 'Codigo': 'EMLG', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736176909&idp=1254735976597'}, {'Id': 303, 'Cod_IOE': '30187', 'Nombre': 'Encuesta Trimestral de Coste Laboral (ETCL)', 'Codigo': 'ETCL', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736045053&idp=1254735976596'}]}
```
</details>

**✅ Buscar operaciones (población)** (1070ms)

- **Parámetros:** `{"query": "poblacion"}`
- **Resultado:** Keys: ['query', 'total_operations', 'operations']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'query': 'poblacion', 'total_operations': 2, 'operations': [{'Id': 35, 'Cod_IOE': '', 'Nombre': 'Poblaciones de hecho desde 1900 hasta 1991. Cifras oficiales sacadas de los Censos respectivos.', 'Codigo': 'DPOH'}, {'Id': 36, 'Cod_IOE': '', 'Nombre': 'Poblaciones de derecho desde 1986 hasta 1995. Cifras oficiales sacadas del Padrón.', 'Codigo': 'DPOD'}]}
```
</details>

**✅ Listar todas las operaciones** (1172ms)

- **Parámetros:** `{"page_size": 10}`
- **Resultado:** Keys: ['query', 'total_operations', 'operations']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'query': None, 'total_operations': 110, 'operations': [{'Id': 4, 'Cod_IOE': '30147', 'Nombre': 'Estadística de Efectos de Comercio Impagados', 'Codigo': 'EI'}, {'Id': 6, 'Cod_IOE': '30211', 'Nombre': 'Índice de Coste Laboral Armonizado', 'Codigo': 'ICLA', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736053992&idp=1254735976596'}, {'Id': 7, 'Cod_IOE': '30168', 'Nombre': 'Estadística de Transmisión de Derechos de la Propiedad', 'Codigo': 'ETDP', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736171438&idp=1254735576606'}, {'Id': 10, 'Cod_IOE': '30256', 'Nombre': 'Indicadores Urbanos', 'Codigo': 'UA', 'Url': 'https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176957&idp=1254735976608'}, {'Id': 13, 'Cod_IOE': '30219', 'Nombre': 'Estadística del Procedimiento Concursal', 'Codigo': 'EPC', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736177018&idp=1254735576606'}, {'Id': 14, 'Cod_IOE': '30182', 'Nombre': 'Índices de Precios del Sector Servicios', 'Codigo': 'IPS', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736176864&idp=1254735576778'}, {'Id': 15, 'Cod_IOE': '30457', 'Nombre': 'Índice de Precios de la Vivienda (IPV)', 'Codigo': 'IPV', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736152838&idp=1254735976607'}, {'Id': 16, 'Cod_IOE': '', 'Nombre': 'Distribución de Nombres', 'Codigo': 'TNOM', 'Url': 'https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736177009&idp=1254735572981'}, {'Id': 18, 'Cod_IOE': '30180', 'Nombre': 'Índice de Precios de Consumo Armonizado (IPCA)', 'Codigo': 'IPCA', 'Url': '/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736176803&idp=1254735976607'}, {'Id': 20, 'Cod_IOE': '30013', 'Nombre': 'Contabilidad Nacional Trimestral de España. Base 2000', 'Codigo': 'CNTR2000'}]}
```
</details>

**✅ Obtener tablas de operación (IPC)** (1347ms)

- **Parámetros:** `{"operation_id": "25"}`
- **Resultado:** Keys: ['operation_id', 'total_tables', 'tables']

**✅ Operación sin tablas (respuesta vacía)** (71ms)

- **Parámetros:** `{"operation_id": "99999"}`
- **Resultado:** Keys: ['operation_id', 'total_tables', 'tables']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'operation_id': '99999', 'total_tables': 0, 'tables': []}
```
</details>


#### `ine_download`

**✅ Descargar datos de tabla IPC** (80ms)

- **Parámetros:** `{"table_id": "50902", "n_last": 5}`
- **Resultado:** Keys: ['table_id', 'n_last', 'total_records', 'data']

**✅ Descargar más períodos** (118ms)

- **Parámetros:** `{"table_id": "50902", "n_last": 12}`
- **Resultado:** Keys: ['table_id', 'n_last', 'total_records', 'data']

**✅ Tabla inexistente (error esperado)** (318ms)

- **Parámetros:** `{"table_id": "99999999", "n_last": 5}`
- **Resultado:** Expected error: INEClientError

<details>
<summary>📄 Respuesta completa</summary>

```json
INEClientError: HTTP 404: <!DOCTYPE html>
<html lang="es">
<head>
<title>404</title>


<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta http-equiv="content-script-type" content="text/javascript" >
<meta http
```
</details>


### 🌤️ AEMET

#### `aemet_get_forecast`

**✅ Pronóstico Madrid (28079)** (360ms)

- **Parámetros:** `{"municipality_code": "28079"}`
- **Resultado:** List with 1 items

**✅ Pronóstico Barcelona (08019)** (321ms)

- **Parámetros:** `{"municipality_code": "08019"}`
- **Resultado:** List with 1 items

**✅ Municipio inexistente (error esperado)** (134ms)

- **Parámetros:** `{"municipality_code": "99999"}`
- **Resultado:** Expected error: AEMETClientError

<details>
<summary>📄 Respuesta completa</summary>

```json
AEMETClientError: Error al obtener los datos
```
</details>


#### `aemet_get_observations`

**✅ Observaciones todas las estaciones** (1322ms)

- **Parámetros:** `{}`
- **Resultado:** List with 9354 items

**✅ Observaciones Madrid-Retiro (3129)** (291ms)

- **Parámetros:** `{"station_id": "3129"}`
- **Resultado:** List with 12 items


#### `aemet_list_locations`

**❌ Listar municipios** (127ms)

- **Parámetros:** `{}`
- **Resultado:** Exception: AEMETClientError

<details>
<summary>❌ Error</summary>

```
AEMETClientError: HTTP 429: {
  "descripcion" : "L�mite de peticiones o caudal por minuto excedido para este usuario. Espere al siguiente minuto.",
  "estado" : 429
}
Traceback (most recent call last):
  File "/home/ubuntu/datos-gob-es-mcp/core/http.py", line 250, in request
    response.raise_for_status()
  File "/home/ubuntu/datos-gob-es-mcp/.venv/lib/python3.12/site-packages/httpx/_models.py", line 829, in raise_for_status
    raise HTTPStatusError(message, request=request, response=self)
httpx.HTTPStatusError: Client error '429 Too Many Requests' for url 'https://opendata.aemet.es/opendata/api/maestro/municipios'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ubuntu/datos-gob-es-mcp/integrations/aemet.py", line 74, in _request
    response = await self.http.get(endpoint, params=params, headers=headers)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/core/http.py", line 347, in get
    return await self.request(
           ^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/core/http.py", line 252, in request
    raise HTTPClientError(
core.http.HTTPClientError: HTTP 429: {
  "descripcion" : "L�mite de peticiones o caudal por minuto excedido para este usuario. Espere al siguiente minuto.",
  "estado" : 429
}

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ubuntu/datos-gob-es-mcp/scripts/test_tools_report.py", line 78, in run_test
    result = await func(**params)
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/scripts/test_tools_report.py", line 594, in _aemet_municipalities_wrapper
    return await aemet_client.get_municipalities()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/integrations/aemet.py", line 106, in get_municipalities
    return await self._request("maestro/municipios")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/integrations/aemet.py", line 101, in _request
    raise AEMETClientError(str(e), status_code=e.status_code) from e
core.exceptions.AEMETClientError: HTTP 429: {
  "descripcion" : "L�mite de peticiones o caudal por minuto excedido para este usuario. Espere al siguiente minuto.",
  "estado" : 429
}

```
</details>

**❌ Listar estaciones** (129ms)

- **Parámetros:** `{}`
- **Resultado:** Exception: AEMETClientError

<details>
<summary>❌ Error</summary>

```
AEMETClientError: HTTP 429: {
  "descripcion" : "L�mite de peticiones o caudal por minuto excedido para este usuario. Espere al siguiente minuto.",
  "estado" : 429
}
Traceback (most recent call last):
  File "/home/ubuntu/datos-gob-es-mcp/core/http.py", line 250, in request
    response.raise_for_status()
  File "/home/ubuntu/datos-gob-es-mcp/.venv/lib/python3.12/site-packages/httpx/_models.py", line 829, in raise_for_status
    raise HTTPStatusError(message, request=request, response=self)
httpx.HTTPStatusError: Client error '429 Too Many Requests' for url 'https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ubuntu/datos-gob-es-mcp/integrations/aemet.py", line 74, in _request
    response = await self.http.get(endpoint, params=params, headers=headers)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/core/http.py", line 347, in get
    return await self.request(
           ^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/core/http.py", line 252, in request
    raise HTTPClientError(
core.http.HTTPClientError: HTTP 429: {
  "descripcion" : "L�mite de peticiones o caudal por minuto excedido para este usuario. Espere al siguiente minuto.",
  "estado" : 429
}

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ubuntu/datos-gob-es-mcp/scripts/test_tools_report.py", line 78, in run_test
    result = await func(**params)
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/scripts/test_tools_report.py", line 598, in _aemet_stations_wrapper
    return await aemet_client.get_stations()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/integrations/aemet.py", line 126, in get_stations
    return await self._request("valores/climatologicos/inventarioestaciones/todasestaciones")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/integrations/aemet.py", line 101, in _request
    raise AEMETClientError(str(e), status_code=e.status_code) from e
core.exceptions.AEMETClientError: HTTP 429: {
  "descripcion" : "L�mite de peticiones o caudal por minuto excedido para este usuario. Espere al siguiente minuto.",
  "estado" : 429
}

```
</details>


### 📜 BOE

#### `boe_get_summary`

**✅ BOE más reciente** (380ms)

- **Parámetros:** `{}`
- **Resultado:** Keys: ['fecha', 'numero', 'secciones', 'note']

**✅ BOE fecha específica (20260108)** (80ms)

- **Parámetros:** `{"date": "20260108"}`
- **Resultado:** Keys: ['fecha', 'numero', 'secciones']


#### `boe_get_document`

**❌ Obtener documento (BOE-A-2026-558)** (70ms)

- **Parámetros:** `{"document_id": "BOE-A-2026-558"}`
- **Resultado:** Exception: BOEClientError

<details>
<summary>❌ Error</summary>

```
BOEClientError: HTTP 404: <?xml version="1.0" encoding="utf-8"?>
<response>
  <status>
    <code>404</code>
    <text>No se ha localizado la operación requerida.</text>
  </status>
  <data/>
</response>

Traceback (most recent call last):
  File "/home/ubuntu/datos-gob-es-mcp/core/http.py", line 250, in request
    response.raise_for_status()
  File "/home/ubuntu/datos-gob-es-mcp/.venv/lib/python3.12/site-packages/httpx/_models.py", line 829, in raise_for_status
    raise HTTPStatusError(message, request=request, response=self)
httpx.HTTPStatusError: Client error '404 Not Found' for url 'https://www.boe.es/datosabiertos/api/boe/documento/BOE-A-2026-558'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ubuntu/datos-gob-es-mcp/integrations/boe.py", line 41, in _request
    response = await self.http.get(endpoint, params=params, headers=headers)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/core/http.py", line 347, in get
    return await self.request(
           ^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/core/http.py", line 252, in request
    raise HTTPClientError(
core.http.HTTPClientError: HTTP 404: <?xml version="1.0" encoding="utf-8"?>
<response>
  <status>
    <code>404</code>
    <text>No se ha localizado la operación requerida.</text>
  </status>
  <data/>
</response>


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ubuntu/datos-gob-es-mcp/scripts/test_tools_report.py", line 78, in run_test
    result = await func(**params)
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/scripts/test_tools_report.py", line 743, in _boe_document_wrapper
    data = await boe_client.get_document(document_id)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/integrations/boe.py", line 67, in get_document
    return await self._request(f"boe/documento/{document_id}")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/datos-gob-es-mcp/integrations/boe.py", line 50, in _request
    raise BOEClientError(str(e), status_code=e.status_code) from e
core.exceptions.BOEClientError: HTTP 404: <?xml version="1.0" encoding="utf-8"?>
<response>
  <status>
    <code>404</code>
    <text>No se ha localizado la operación requerida.</text>
  </status>
  <data/>
</response>


```
</details>

**✅ Documento inexistente (error esperado)** (68ms)

- **Parámetros:** `{"document_id": "BOE-X-9999-99999"}`
- **Resultado:** Expected error: BOEClientError

<details>
<summary>📄 Respuesta completa</summary>

```json
BOEClientError: HTTP 404: <?xml version="1.0" encoding="utf-8"?>
<response>
  <status>
    <code>404</code>
    <text>No se ha localizado la operación requerida.</text>
  </status>
  <data/>
</response>

```
</details>


#### `boe_search`

**✅ Búsqueda básica (subvenciones)** (569ms)

- **Parámetros:** `{"query": "subvenciones"}`
- **Resultado:** Keys: ['query', 'total_results', 'results']

<details>
<summary>📄 Respuesta completa</summary>

```json
{'query': 'subvenciones', 'total_results': 2, 'results': [{'id': 'BOE-A-2026-604', 'titulo': 'Resolución de 23 de diciembre de 2025, de la Subsecretaría, por la que se publica el Convenio entre la Entidad Estatal de Seguros Agrarios, O.A., y la Xunta de Galicia, para el intercambio de datos a efectos de gestión y control de subvenciones a los seguros agrarios de los beneficiarios de su comunidad autónoma.', 'fecha': '20260110'}, {'id': 'BOE-A-2026-605', 'titulo': 'Orden TMD/1586/2025, de 29 de diciembre, por la que se modifica la Orden TMD/101/2025, de 31 de enero, para la concesión directa de subvenciones a ayuntamientos y diputaciones provinciales para financiar obras de reparación, restitución o reconstrucción de infraestructuras, equipamientos o instalaciones y servicios de titularidad municipal y de la red viaria de titularidad provincial, de los daños causados por la Depresión Aislada en Niveles Altos (DANA) entre el 28 de octubre y el 4 de noviembre de 2024, al amparo del artículo 5 del Real Decreto-ley 6/2024, de 5 de noviembre.', 'fecha': '20260110'}]}
```
</details>

**✅ Búsqueda con fechas** (841ms)

- **Parámetros:** `{"query": "educación", "date_from": "20240101", "date_to": "20240331"}`
- **Resultado:** Keys: ['query', 'total_results', 'results']


## 📈 Estadísticas de Latencia

| Herramienta | Min (ms) | Max (ms) | Promedio (ms) | Tests |
|-------------|----------|----------|---------------|-------|
| `aemet_get_forecast` | 134 | 360 | 272 | 3 |
| `aemet_get_observations` | 291 | 1322 | 806 | 2 |
| `aemet_list_locations` | 127 | 129 | 128 | 2 |
| `boe_get_document` | 68 | 70 | 69 | 2 |
| `boe_get_summary` | 80 | 380 | 230 | 2 |
| `boe_search` | 569 | 841 | 705 | 2 |
| `get` | 50 | 62 | 55 | 3 |
| `ine_download` | 80 | 318 | 172 | 3 |
| `ine_search` | 71 | 1347 | 998 | 5 |
| `search` | 86 | 151 | 115 | 4 |

### ❌ Errores Encontrados

- **aemet_list_locations** - Listar municipios: Exception: AEMETClientError
- **aemet_list_locations** - Listar estaciones: Exception: AEMETClientError
- **boe_get_document** - Obtener documento (BOE-A-2026-558): Exception: BOEClientError

## 🔍 Detalles Técnicos

| Detalle | Valor |
|---------|-------|
| Python | 3.12.3 |
| Sistema | Linux 6.14.0-1017-aws |
| Fecha | 2026-01-11 15:02:25 |
| Timezone | UTC |

---

*Este reporte fue generado automáticamente por `scripts/test_tools_report.py`*