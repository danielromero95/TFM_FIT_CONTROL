# FIT CONTROL

**Repositorio:** `TFM_FIT_CONTROL`
**Autor:** Daniel Romero
**Version:** 3.1 (22-12-2025)

---

## Descripción

Este proyecto ofrece un sistema de análisis de ejercicios impulsado por visión por computador.
A partir de vídeos de entrenamiento de fuerza (sentadilla, press de banca, peso muerto), la aplicación busca:

- Detectar el tipo de ejercicio (sentadilla / press de banca / peso muerto) y la vista de cámara (frontal / lateral) con puntuaciones de confianza.
- Contar repeticiones automáticamente.
- Calcular ángulos articulares clave (rodilla, cadera, hombro).
- Detectar posibles errores de técnica (desarrollo futuro).
- Proporcionar una demo web sencilla para visualizar y descargar los resultados.

El flujo principal incluye:

- **Preprocesamiento de vídeo:** extracción de fotogramas, redimensionado, normalización, filtrado (Gaussian, CLAHE) y recorte de ROI.
- **Estimación de pose:** MediaPipe Pose (BlazePose 3D) para extraer landmarks (33 puntos por frame) y calcular ángulos / velocidades angulares / simetrías.
- **Análisis:** detección de repeticiones, agregación de métricas y (por desarrollar) alertas sobre desviaciones técnicas.

---

## Instalación y ejecución

El proyecto incluye los archivos `pyproject.toml` y `requirements.txt` para la creación del entorno virtual con las dependencias necesarias para la ejecución del proyecto.

### Opción A — uv (recomendada)

~~~
# Desde la raíz del repositorio
uv venv
uv pip install -e .
uv run streamlit run src/app.py
~~~

### Opción B — venv + pip

~~~
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -e .
streamlit run src/app.py
~~~

## Pasos en la interfaz

1. Sube un vídeo de entrenamiento.
2. Detecta ejercicio y vista (automático o manual).
3. Configura parámetros (FPS, Modelo de MediaPipe, rotación, vídeo de depuración, etc.).
4. Ejecuta el análisis (asíncrono; se muestra el progreso).
5. Resultados: conteo y velocidad de repeticiones, métricas, vídeo de depuración opcional y descargas.

---
## Novedades en 3.1

- **Visualización de velocidad de repetición.** Los resultados incluyen una gráfica y tabla de cadencia
  por repetición para contextualizar el ritmo del atleta y detectar ralentizaciones o aceleraciones
  entre repeticiones.
- **Controles de rendimiento y avisos de FPS.** La configuración permite ajustar la complejidad del
  modelo de pose y los FPS objetivo con ayudas descriptivas multilínea; la pipeline advierte sobre
  vídeos de bajo FPS sin descartar repeticiones si puede contarlas con fiabilidad.
- **Depuración y descargas simplificadas.** El informe de depuración combina métricas, configuración
  efectiva y overlays reducidos en un solo paquete descargable con nombres de sesión estampados en el
  tiempo, eliminando columnas redundantes y archivos auxiliares innecesarios.
- **UI más compacta y coherente.** Se centran el título y los parámetros de ejecución, se reduce el
  espaciado entre bloques, se ajustan los botones y se modernizan los componentes Streamlit para
  mantener la compatibilidad con las APIs actuales.


## Novedades en 3.0

- **Protecciones para medios pesados.** La pipeline detecta vídeos grandes y
  reduce la previsualización o escala/desactiva los overlays para evitar cuellos
  de botella en máquinas modestas. También limita los FPS de depuración para
  que la reproducción se mantenga fluida incluso en clips densos.
- **Muestreo temporal y overlays web-safe.** El muestreo se recalibra en función
  de los metadatos (o el lector cuando son poco fiables) para trabajar con FPS
  efectivos coherentes, y los vídeos de depuración se generan con copias H.264
  listas para reproducirse en navegador.
- **Clasificación de ejercicio más transparente.** Las heurísticas que puntúan
  cada levantamiento (profundidad, recorrido de la barra, simetría de brazos,
  inclinación del torso) se documentan y priorizan señales más fiables para
  distinguir sentadilla, peso muerto y press de banca.

## Novedades en 2.8

- **Selección más inteligente del ángulo principal.** La pipeline de análisis ahora elige el ángulo articular más completo para el ejercicio detectado, ajusta automáticamente los umbrales de prominencia/distancia y exporta la configuración efectiva de ejecución para que el conteo de repeticiones coincida con el movimiento capturado.
- **Cobertura de métricas ampliada.** Se calculan los ángulos de bisagra de cadera e inclinación del tronco junto con las métricas existentes de rodilla y codo, y se exponen como valores predeterminados sensibles al ejercicio al graficar resultados.
- **Guía contextual de métricas.** El selector de métricas de resultados resalta el ángulo principal de conteo, reinicia selecciones por ejecución y añade popovers de ayuda en línea que explican cada métrica y cómo se relaciona con el conteo de repeticiones.
- **Overlays visuales de umbrales.** El visor sincronizado de métricas puede mostrar los umbrales configurados bajos/altos junto con las bandas de repeticiones manteniendo la leyenda en un diseño compacto bajo la gráfica.

## Novedades en 2.7

- La rotación de vídeo ahora se corrige por completo en toda la pipeline, evitando overlays invertidos y asegurando que los vídeos de depuración exportados respeten la orientación prevista.
- La visualización de métricas permanece sincronizada con la reproducción: el cursor, los overlays y el diseño de la gráfica se refinaron para que las interacciones sigan siendo precisas por fotograma incluso tras relayouts.
- La sincronización con BroadcastChannel enlaza la vista de detección y el visor de métricas, manteniendo múltiples componentes en sintonía durante la reproducción.
- Los ajustes de overlays conscientes de landmarks mantienen rotaciones coherentes al combinar landmarks renderizados con fotogramas capturados.

## Novedades en 2.6

- Fuente única de verdad para el conteo de repeticiones.

- Nuevo paquete: src/C_analysis/ con repetition_counter.py y el resto de utilidades de análisis.

- Eliminado el paquete legado src/D_modeling/ y el archivo sin uso fault_detection.py.

- Contador de repeticiones robusto basado en valles.

- Detección de valles sobre la secuencia de ángulos invertida con umbrales de prominencia y distancia mínima.

- Consolidación por ventana refractaria: agrupa valles cercanos y mantiene el más prominente por ventana.

- Salida de depuración alineada: CountingDebugInfo(valley_indices, prominences) se mantiene consistente tras el filtrado.



## Novedades en 2.5

- Visor sincronizado de vídeo + métricas (Plotly). Un cursor preciso por fotograma se vincula a la reproducción; al hacer clic en la gráfica se busca el vídeo a ese punto exacto.

- Conciencia de repeticiones. Bandas sombreadas marcan cada repetición y un slider “Go to rep” salta la reproducción al inicio de la repetición seleccionada.

- Renderizado rápido y fluido. El downsampling inteligente (límite ≈3k puntos) mantiene las gráficas ágiles mientras requestAnimationFrame ofrece actualizaciones suaves del cursor.

- Mejor navegación en las gráficas. Zoom con scroll y doble clic para reiniciar el eje X; tooltips de hover unificados en el eje X.

- Compatibilidad robusta. Funciona con Streamlit 1.50 (manejo flexible cuando components.html no soporta key).

- Helper de vídeo reutilizable. El método público get_video_source expone la lógica existente de data-URI para otros módulos.

- Mejoras de UX en resultados. La selección de métricas se persiste, frame_idx se oculta en el selector y la app recurre con gracia al renderizador de vídeo legado cuando las métricas no están disponibles o no se seleccionan.

- Arquitectura modular. El nuevo componente src/ui/metrics_sync.py encapsula el visor y puede reutilizarse en distintas pantallas.


## Novedades en 2.3

**Actualización de documentación y estructura del proyecto.** Esta versión se centra en mejorar la descubribilidad y alinear la estructura del repositorio con los módulos refactorizados:

- README actualizado con la información más reciente y secciones de changelog reorganizadas.
- Añade una vista general de alto nivel de la estructura actual del proyecto para acelerar la incorporación.
- Se clarifica el propósito de los recursos legados almacenados en `OLD/` como referencia histórica.

## Novedades en 2.0

**Detección automática de ejercicio y vista (MVP)**

- Añadido `src/exercise_detection/exercise_detector.py` (y `src/exercise_detection/__init__.py`).
- Detecta sentadilla / press de banca / peso muerto + vista frontal / lateral con una puntuación de confianza.
- Usa landmarks de MediaPipe con heurísticas (ROMs, desplazamiento de pelvis, recorrido de muñeca) y nuevas señales de vista:
  - Yaw de hombros (ángulo 3D a partir de la profundidad de hombro izquierdo/derecho).
  - Delta de profundidad z de hombros (asimetría).
  - Anchura de hombros normalizada (anchura/longitud de torso) con comprobación de estabilidad.

**Integración con Streamlit**

- El botón “Detectar ejercicio (beta)” ahora ejecuta la detección, cachea resultados y actualiza el selectbox.
- Los resultados de detección (etiqueta, vista, confianza) se muestran en la UI.
- `run_pipeline(..., prefetched_detection=...)` acepta resultados cacheados de la UI para evitar ejecuciones duplicadas del detector.

**RunStats ahora incluye:**

- `exercise_selected` (elección de la UI)
- `exercise_detected`, `view_detected`, `detection_confidence`
- La tabla de resultados muestra seleccionado vs detectado para mayor transparencia.

**Depuración y tests**

- El logging ligero de depuración imprime una única línea `DET DEBUG` con estadísticas cinemáticas y de vista (ROMs, media/std de anchura, mediana de yaw, mediana de profundidad z).
- Fixtures sintéticos para sentadilla/press de banca/peso muerto y comprobaciones de vista frontal vs lateral.

## Novedades en 1.1

**Paridad lograda entre Streamlit y la UI de escritorio.** Ambos front-ends ahora llaman a la misma pipeline unificada y configuración:

- `Config` unificada (dataclasses) con **fingerprint** SHA1 para reproducibilidad.
- Punto de entrada único `run_pipeline(video_path, cfg)` que devuelve un `Report` completo.
- Detección de FPS robusta con alternativas y salvaguardas (`min_frames`, `min_fps`).
- Una única etapa de redimensionado (documentada en la config) antes de la estimación de pose.
- Conteo de repeticiones basado en valles con **prominence / min distance / refractory**.
- Streamlit y la interfaz de escritorio exponen **run stats**, avisos y `skip_reason`.
- OpenCV/SciPy fijados en `environment.yml` para mayor estabilidad.

---

## Estructura del proyecto (v3.0)

| Ruta | Descripción |
| --- | --- |
| `src/app.py` | Punto de entrada de Streamlit que conecta la UI con la pipeline de análisis. |
| `src/A_preprocessing/` | Extracción de frames y utilidades de vídeo (metadatos, saneo, recorte) previas a la estimación de pose. |
| `src/B_pose_estimation/` | Pipeline de pose: constantes, estimadores, geometría y métricas derivadas a partir de landmarks. |
| `src/C_analysis/` | Análisis posterior: conteo de repeticiones, métricas agregadas, overlays y streaming de resultados. |
| `src/D_visualization/` | Renderizado de landmarks, estilos de overlay y escritura de vídeos de depuración. |
| `src/config/` | Dataclasses de configuración, valores por defecto y validaciones para ejecutar la pipeline. |
| `src/core/` | Utilidades compartidas (I/O, matemáticas, logging) empleadas por varios módulos. |
| `src/exercise_detection/` | Heurísticas para clasificar ejercicio y vista de cámara a partir de señales cinemáticas. |
| `src/ui/` | Componentes específicos de Streamlit (sincronización de métricas, helpers de vídeo). |
| `src/pipeline_data.py` | Tipos y estructuras de datos comunes que comparten las etapas de la pipeline. |
| `tests/` | Tests automatizados (unitarios e integración) para la pipeline y utilidades clave. |
| `docs/` | Anteproyecto de Fit Control. |
| `1-ENTREGA_TFM/` | Material de la entrega (memoria, tablas e imágenes de soporte). |
| `pyproject.toml` | Dependencias fijadas para entornos que usan `uv`. |
| `requirements.txt` | Dependencias fijadas para entornos que no usan `uv`. |
| `uv.lock` | Versiones bloqueadas para instalaciones reproducibles con uv. |
| `project_tree.cmd` | Script de ayuda para regenerar un árbol resumido del repositorio. |

## Notas para desarrolladores: ajuste del clasificador de ejercicios

Las heurísticas implementadas en `src/exercise_detection` exponen todos los umbrales
ajustables en `constants.py`, de modo que se pueden modificar sin tocar los módulos de
lógica:

- **Sampling y smoothing**: `DEFAULT_SAMPLING_RATE`, `SMOOTHING_*`.
- **Segmentación**: histéresis de rodilla y margen de caída de la barra.
- **Clasificador de vista**: pesos por feature y márgenes de decisión.
- **Puntuaciones de ejercicio**: ángulos de corte, penalizaciones y multiplicadores de veto.

La pipeline de clasificación emite un log compacto `INFO` por clip con los
valores agregados y las puntuaciones brutas/ajustadas. El log `DEBUG` incluye los
fondos por repetición. Son la forma más rápida de comprobar qué señales dominan
una predicción al calibrar las constantes.

Ejecuta los tests sintéticos con `pytest -q` para confirmar que los refactors
mantienen el comportamiento cualitativo esperado en sentadilla, peso muerto,
press de banca y escenarios desconocidos.

Al añadir nuevas métricas derivadas o estadísticas, utiliza los helpers de
`exercise_detection.stats` (por ejemplo `safe_nanmedian` y `safe_nanstd`).
Protegen frente a entradas llenas de NaN para evitar `RuntimeWarning` de NumPy
mientras se preserva el comportamiento actual.

---
