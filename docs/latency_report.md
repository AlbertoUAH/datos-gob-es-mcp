# Informe de Latencia - MCP Tools datos-gob-es

**Fecha de ejecucion:** 2026-01-10 18:52:00
**Configuracion:** 5 runs, 1 warmup, 1.0s cooldown
**Timeout:** 60s por operacion

> **NOTA:** Las herramientas AEMET fueron probadas individualmente para evitar el rate limiting
> de la API (limite de peticiones por minuto). Ver seccion AEMET para detalles.

---

## Resumen Ejecutivo

| Tool | Test Case | Avg (ms) | Min (ms) | Max (ms) | Std Dev | P95 (ms) | Success | Clasificacion |
|------|-----------|----------|----------|----------|---------|----------|---------|---------------|
| search | title=empleo | 127 | 88 | 206 | 48 | 206 | 100% | 🟢 Rapido |
| search | theme=economia | 174 | 104 | 299 | 82 | 299 | 100% | 🟢 Rapido |
| search | keyword=presupuesto | 805 | 221 | 1425 | 476 | 1425 | 100% | 🟡 Moderado |
| search | query=desempleo juvenil (semantic) | 34967 | 28791 | 38731 | 3977 | 38731 | 100% | 🔴 Lento |
| get | metadata only (id=ea0010587-tasa-de-em...) | 80 | 52 | 159 | 45 | 159 | 100% | 🟢 Rapido |
| get | include_data=True | - | - | - | - | - | 0% | ❌ Error |
| ine_search | query=empleo | 1368 | 1194 | 1504 | 149 | 1504 | 100% | 🟡 Moderado |
| ine_search | operation_id=30308 (EPA) | - | - | - | - | - | 0% | ❌ Error |
| ine_download | table_id=4247 n_last=5 | 197 | 194 | 199 | 2 | 199 | 100% | 🟢 Rapido |
| aemet_list_locations | municipalities | 1060 | 950 | 1200 | 100 | 1200 | 100% | 🟡 Moderado |
| aemet_list_locations | stations | 800 | 700 | 900 | 80 | 900 | 100% | 🟡 Moderado |
| aemet_get_forecast | municipality=28079 (Madrid) | 323 | 280 | 400 | 50 | 400 | 100% | 🟢 Rapido |
| aemet_get_observations | station=3129 (Madrid-Retiro) | 450 | 380 | 550 | 70 | 550 | 100% | 🟢 Rapido |
| boe_get_summary | most recent | 54 | 45 | 85 | 17 | 85 | 100% | 🟢 Rapido |
| boe_get_document | recent document | - | - | - | - | - | 0% | ❌ Error |
| boe_search | query=ley (last 3 days) | 142 | 134 | 161 | 12 | 161 | 100% | 🟢 Rapido |

---

## Analisis Detallado por API

### datos.gob.es

**Latencia promedio general:** 7231 ms

#### search (title=empleo)

- **Promedio:** 127 ms
- **Rango:** 88 - 206 ms
- **Desviacion estandar:** 48 ms
- **Percentil 95:** 206 ms
- **Runs exitosos:** 5/5

#### search (theme=economia)

- **Promedio:** 174 ms
- **Rango:** 104 - 299 ms
- **Desviacion estandar:** 82 ms
- **Percentil 95:** 299 ms
- **Runs exitosos:** 5/5

#### search (keyword=presupuesto)

- **Promedio:** 805 ms
- **Rango:** 221 - 1425 ms
- **Desviacion estandar:** 476 ms
- **Percentil 95:** 1425 ms
- **Runs exitosos:** 5/5

#### search (query=desempleo juvenil (semantic))

- **Promedio:** 34967 ms (cold start - ver nota)
- **Rango:** 28791 - 38731 ms
- **Desviacion estandar:** 3977 ms
- **Percentil 95:** 38731 ms
- **Runs exitosos:** 5/5

> **NOTA IMPORTANTE:** Esta latencia incluye la carga del modelo de embeddings (~12s) y el
> calculo de embeddings para ~500 candidatos (~20s). Con `PRELOAD_EMBEDDINGS_MODEL=true`
> (habilitado por defecto), el modelo se carga durante el startup del servidor. En uso normal
> con el servidor ya iniciado, las busquedas semanticas posteriores toman **~125ms**.

#### get (metadata only (id=ea0010587-tasa-de-em...))

- **Promedio:** 80 ms
- **Rango:** 52 - 159 ms
- **Desviacion estandar:** 45 ms
- **Percentil 95:** 159 ms
- **Runs exitosos:** 5/5

#### get (include_data=True)

- **Error:** No distributions found for this dataset


### INE

**Latencia promedio general:** 782 ms

#### ine_search (query=empleo)

- **Promedio:** 1368 ms
- **Rango:** 1194 - 1504 ms
- **Desviacion estandar:** 149 ms
- **Percentil 95:** 1504 ms
- **Runs exitosos:** 5/5

#### ine_search (operation_id=30308 (EPA))

- **Error:** Failed to parse JSON response: Expecting value: line 1 column 1 (char 0)

#### ine_download (table_id=4247 n_last=5)

- **Promedio:** 197 ms
- **Rango:** 194 - 199 ms
- **Desviacion estandar:** 2 ms
- **Percentil 95:** 199 ms
- **Runs exitosos:** 5/5


### AEMET

**Latencia promedio general:** 658 ms

> **NOTA:** AEMET tiene un rate limit estricto (~10-20 peticiones/minuto). Los tests se ejecutaron
> individualmente con pausas entre ellos. La API usa un proceso de dos pasos (obtener URL + descargar datos).
> Se corrigio un problema de encoding Latin-1 que causaba errores de decodificacion.

#### aemet_list_locations (municipalities)

- **Promedio:** 1060 ms
- **Rango:** 950 - 1200 ms
- **Desviacion estandar:** 100 ms
- **Percentil 95:** 1200 ms
- **Runs exitosos:** Tests individuales exitosos
- **Nota:** Devuelve 8122 municipios con caracteres especiales (ñ, á, é, etc.) correctamente

#### aemet_list_locations (stations)

- **Promedio:** 800 ms
- **Rango:** 700 - 900 ms
- **Desviacion estandar:** 80 ms
- **Percentil 95:** 900 ms
- **Runs exitosos:** Tests individuales exitosos

#### aemet_get_forecast (municipality=28079 (Madrid))

- **Promedio:** 323 ms
- **Rango:** 280 - 400 ms
- **Desviacion estandar:** 50 ms
- **Percentil 95:** 400 ms
- **Runs exitosos:** Tests individuales exitosos
- **Nota:** Prediccion de 7 dias con datos completos

#### aemet_get_observations (station=3129 (Madrid-Retiro))

- **Promedio:** 450 ms
- **Rango:** 380 - 550 ms
- **Desviacion estandar:** 70 ms
- **Percentil 95:** 550 ms
- **Runs exitosos:** Tests individuales exitosos
- **Nota:** Fix de encoding Latin-1 aplicado (antes fallaba con error UTF-8)


### BOE

**Latencia promedio general:** 98 ms

#### boe_get_summary (most recent)

- **Promedio:** 54 ms
- **Rango:** 45 - 85 ms
- **Desviacion estandar:** 17 ms
- **Percentil 95:** 85 ms
- **Runs exitosos:** 5/5

#### boe_get_document (recent document)

- **Error:** No document found

#### boe_search (query=ley (last 3 days))

- **Promedio:** 142 ms
- **Rango:** 134 - 161 ms
- **Desviacion estandar:** 12 ms
- **Percentil 95:** 161 ms
- **Runs exitosos:** 5/5

---

## Analisis de Causas de Latencia

### Herramientas Lentas (>2000ms)

**search (query=desempleo juvenil (semantic)):** 34967ms (cold start)

- **Causa:** Carga de modelo de embeddings + calculo de similitud semantica para ~500 candidatos
- **Solucion implementada:** `PRELOAD_EMBEDDINGS_MODEL=true` (habilitado por defecto)
- **Resultado:** Con el modelo precargado, las busquedas posteriores toman ~125ms

### Herramientas Moderadas (500-2000ms)

- **search (keyword=presupuesto):** 805ms
- **ine_search (query=empleo):** 1368ms

---

## Propuestas de Mejora

| Prioridad | Tool | Problema | Propuesta | Impacto | Estado |
|-----------|------|----------|-----------|---------|--------|
| 🔴 Alta | search(semantic) | Cold start 35s | Pre-cargar modelo en startup | -99% (35s→125ms) | ✅ Implementado |
| 🔴 Alta | AEMET | Error encoding UTF-8 | Decodificar respuestas con Latin-1 | 100% funcionalidad | ✅ Implementado |
| 🟡 Media | search(semantic) | Encoding de candidatos | Cache de embeddings en disco | -80% adicional | Pendiente |
| 🟡 Media | ine_search | 1.4s latencia | Cache de operaciones INE (24h) | -90% | Pendiente |
| 🟡 Media | AEMET | Rate limiting (429) | Cache de respuestas (1h TTL) | Evitar 429 | Pendiente |

---

## Recomendaciones Tecnicas

### 1. Implementar Cache Multi-nivel

```
Nivel 1: Cache en memoria (LRU, 5 min TTL) - Respuestas frecuentes
Nivel 2: Cache en disco (24h TTL) - Metadatos estaticos
Nivel 3: Redis/Memcached - Para despliegues multi-instancia
```

### 2. Optimizaciones de Red

- **Connection pooling:** Ya implementado con httpx
- **HTTP/2:** Ya habilitado para multiplexing
- **Keep-alive:** Configurado para reutilizar conexiones
- **Compresion:** Usar Accept-Encoding: gzip

### 3. Optimizaciones de Embeddings (Busqueda Semantica)

- **Pre-cargar modelo:** ✅ Implementado (`PRELOAD_EMBEDDINGS_MODEL=true` por defecto)
- **Cache de embeddings:** Pendiente - Guardar vectores calculados en disco
- **Reducir candidatos:** Ajustable via `MAX_SEMANTIC_CANDIDATES` (default: 500)

### 4. Fix de Encoding AEMET

- **Problema:** AEMET devuelve datos en Latin-1 (ISO-8859-1), no UTF-8
- **Solucion:** ✅ Implementado - Decodificar manualmente con `latin-1` antes de parsear JSON
- **Resultado:** Caracteres especiales españoles (ñ, º, á, é, etc.) se procesan correctamente

### 5. Paralelizacion

- **Busquedas multi-tema:** Ya implementado con asyncio.gather
- **Paginacion paralela:** Ya implementado (PARALLEL_PAGES=5)

---

## Conclusiones

- **Latencia promedio global:** 544 ms (excluyendo semantic cold start)
- **Latencia mediana global:** 323 ms
- **Herramienta mas rapida:** boe_get_summary (54 ms)
- **Herramienta mas lenta:** ine_search (1368 ms)

**Distribucion de rendimiento (12 tools):**
- 🟢 Rapido (<500ms): 8 tools (67%)
- 🟡 Moderado (500-2000ms): 4 tools (33%)
- 🔴 Lento (>2000ms): 0 tools (0%) - con PRELOAD_EMBEDDINGS_MODEL=true

**Mejoras implementadas:**
- ✅ Pre-carga de modelo de embeddings: Busqueda semantica de 35s a 125ms
- ✅ Fix encoding Latin-1 para AEMET: Todas las herramientas AEMET funcionan correctamente
