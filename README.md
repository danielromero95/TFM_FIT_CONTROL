# FIT CONTROL

**Repositorio:** `TFM_FIT_CONTROL`
**Autor:** Daniel Romero
**Versión:** 3.3 (14-01-2026)

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

### Opcional — ffprobe (FFmpeg)

Si está disponible en el sistema, `ffprobe` permite enriquecer los metadatos de vídeo incluidos en el reporte.
Es opcional: la generación de resultados sigue funcionando con la información proporcionada por OpenCV si no está instalado.

## Pasos en la interfaz

1. Sube un vídeo de entrenamiento.
2. Detecta ejercicio y vista (automático o manual).
3. Configura parámetros (FPS, Modelo de MediaPipe, rotación, vídeo de depuración, etc.).
4. Ejecuta el análisis (asíncrono; se muestra el progreso).
5. Resultados: conteo y velocidad de repeticiones, métricas, vídeo de depuración opcional y descargas; el paquete `.zip` incluye
   también `video_data.csv`/`video_data.json` con metadatos del vídeo de entrada junto con la configuración efectiva, estadísticas,
   métricas y resto de artefactos.


## Novedades en 3.3

- **Diagnósticos completos de detección exportables.** La detección de ejercicio/vista ahora publica un bloque de diagnósticos con puntuaciones crudas/ajustadas, penalizaciones, probabilidades, márgenes, gates de deadlift y resumen de vista. Estos datos se guardan en `RunStats` y se exponen en el reporte; además, cuando hay series de depuración de brazo/barra, se exportan a `arm_debug_timeseries.csv` para auditoría visual.
- **Separación más robusta entre sentadilla y peso muerto.** Se añade `bar_above_hip_norm` (altura de la barra respecto a la cadera en la ventana inferior) y se utiliza para premiar sentadilla, penalizar peso muerto y desempatar cuando la barra está claramente alta/baja.
- **Selección de barra/brazo más fiable.** La heurística ahora prioriza proxies con mayor ratio de frames válidos (muñeca/codo/hombro) y evita señales inestables o inconsistentes, reduciendo falsos positivos en clips con tracking parcial.
- **Normalización del eje Y para métricas.** Se detecta si el eje vertical viene invertido y se corrigen automáticamente las series `*_y`, manteniendo coherencia en comparativas y diagnósticos.
- **Sincronización vídeo ↔ métricas endurecida.** Se corrigieron desajustes de timebase y repeticiones, mejorando la alineación de overlays, gráficas y marcas de repetición en toda la UI.

## Novedades en 3.2

- **Reporte ampliado con metadatos del vídeo de entrada.** El paquete descargable ahora incluye `video_data.csv` (formato clave/valor en texto para máxima compatibilidad) y `video_data.json` (preserva tipos como int/float/bool donde aplica) con container/codec, resolución, FPS, duración, rotación, presencia de audio y otros metadatos del input. Ambos incorporan un bloque `preprocessing_decisions` que expone la rotación aplicada, la estrategia de muestreo, el FPS efectivo, los frames analizados y avisos relevantes.
- **Nombre del bundle estable y trazable.** El archivo descargable se denomina `<Nombre_archivo_input>-YYYY_MM_DD-HH_MM.zip`, reutilizando el nombre original subido en lugar de nombres temporales del sistema.
- **Trazabilidad del nombre original en la UI/pipeline.** Se conserva `video_original_name` durante la subida y el análisis para que el reporte y las exportaciones reflejen el nombre real del vídeo.
- **RunStats enriquecido para auditoría.** Se registran campos como `rotation_applied`, `sample_rate`, `sampling_strategy`, `file_size_bytes` y `frame_count_input`, además de los FPS y contadores existentes, para entender las decisiones de rendimiento y muestreo.
- **Metadatos más robustos (OpenCV + ffprobe).** El sistema obtiene metadatos con OpenCV y, cuando está disponible, amplía la información con `ffprobe` de forma tolerante. Nota: `ffprobe` es opcional; si no está instalado, el reporte se genera con los datos disponibles.
- **Gráfica de velocidad por repetición con fases y cálculos corregidos.** La visualización de cadencia ahora desglosa velocidad y duración de fase **Down/Up** (o **Up/Down** en peso muerto), corrige el cálculo de intervalos, FPS y cadencias por repetición, y muestra intervalos y bottoms coherentes con las señales de conteo.
- **Conteo y umbrales afinados según ejercicio.** Se adoptan valores por defecto específicos de sentadilla/peso muerto/press banca, se habilita un único conmutador de activación de filtros, se añade un modo de umbral superior estricto y se reorganiza la UI con un layout más compacto y un bloque dedicado a depuración/configuración avanzada.
- **Artefactos de depuración en las descargas.** El `.zip` incluye `rep_intervals.csv`, `rep_speeds.csv`, `rep_speed_long.csv` y `rep_speed_meta.json` para auditar intervalos, fases y velocidades usadas en la gráfica y en las decisiones de conteo.

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
- `reps_detected_raw`, `reps_detected_final`, `reps_rejected_threshold` para diferenciar repeticiones candidatas y válidas tras el filtrado por umbrales.

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

## Estructura del proyecto

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

Consulta `Exercise_Detection.txt` para una explicación completa y end‑to‑end de la detección automática del ejercicio (extracción, segmentación, métricas, clasificación y diagnósticos).

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

Detalles clave para auditar y ajustar la detección:

- **Diagnósticos serializables.** El clasificador devuelve `raw_scores`, `penalties`, `adjusted_scores`, `probabilities`, `margin`, `tiebreak` y señales específicas como `deadlift_veto`, `bar_above_hip_norm`, `wrist_shoulder_diff_norm` o `wrist_hip_diff_norm`. La clasificación de vista añade un resumen con `lateral_score`, `reliable_frames` y `dispersion`. Todos estos campos se propagan a `RunStats.detection_diagnostics` y, si existe `arm_debug_timeseries`, se exporta a `arm_debug_timeseries.csv` en el reporte para inspección manual.
- **Métrica de barra vs cadera.** `bar_above_hip_norm` se calcula en la ventana inferior como la diferencia normalizada entre la cadera media y la barra (`bar_y`). Esta señal alimenta gates de deadlift (clamps) y bonos/penalizaciones de sentadilla, además de los desempates.
- **Selección de proxy de barra/brazo.** `bar_y` se elige entre muñecas/codos/hombros según el ratio de frames válidos; `arm_y` prefiere muñeca/codo pero evita proxies inestables o inconsistentes con la altura de la cadera. Esto reduce ruido cuando hay landmarks incompletos.
- **Normalización de eje Y.** Si el eje vertical viene invertido (hombros por debajo de cadera en la mediana), las series `*_y` se invierten antes del análisis y se registra `y_axis_flipped` en los diagnósticos.
- **Confianza combinada con vista.** Cuando la vista es conocida, la confianza final del ejercicio se suaviza con la fiabilidad de la vista (`confidence * (0.6 + 0.4 * view_conf)`), evitando el efecto “cap” cuando la vista es “unknown”.

Ejecuta los tests sintéticos con `pytest -q` para confirmar que los refactors
mantienen el comportamiento cualitativo esperado en sentadilla, peso muerto,
press de banca y escenarios desconocidos.

Al añadir nuevas métricas derivadas o estadísticas, utiliza los helpers de
`exercise_detection.stats` (por ejemplo `safe_nanmedian` y `safe_nanstd`).
Protegen frente a entradas llenas de NaN para evitar `RuntimeWarning` de NumPy
mientras se preserva el comportamiento actual.

---

## Notas para desarrolladores: `model_complexity` en MediaPipe Pose

Resumen de findings para el parámetro `model_complexity` que controla el tamaño
del modelo de MediaPipe Pose y su impacto en la pipeline principal:

- **El selector de la UI llega al pipeline principal.** El valor elegido en la
  UI se guarda en `cfg.pose.model_complexity` y la creación de
  `PoseEstimator/CroppedPoseEstimator/RoiPoseEstimator` pasa
  `model_complexity=cfg.pose.model_complexity`, garantizando que cada run use el
  modelo seleccionado y evitando resultados idénticos entre 0/1/2.
- **Rastro de dónde se define y cómo fluye el valor.** Se define en
  `MODEL_COMPLEXITY` / `POSE_MODEL_COMPLEXITY` (`src/config/settings.py`), se
  expone en `PoseConfig` (`src/config/models.py`), se ajusta en la UI con
  `configure_values` y llega a `cfg.pose.model_complexity` en la preparación de
  pipeline (`src/ui/steps/utils/pipeline.py`).
- **Instrumentación para validar el modelo activo.** `PoseGraphPool.acquire()`
  registra en `INFO` cuando crea instancias nuevas e imprime `run_id`,
  `model_complexity` y `static_image_mode`. Esto permite comprobar en consola
  que los runs 0/1/2 instancian modelos distintos.
- **Comportamiento del fallback.** Si `enable_recovery_pass` está activo,
  `_process_with_recovery()` fuerza `model_complexity=2` en el fallback; así que
  incluso un run con 0/1 puede elevarse a 2 cuando se pierde el tracking.
- **Pipelines alternativos no consumen `cfg.pose.model_complexity`.** La
  detección de ejercicio en `src/exercise_detection/extraction.py` e
  `incremental.py` construye `mp_pose.Pose` con `MODEL_COMPLEXITY` fijo (2); si se
  desea coherencia total, habría que inyectar el valor configurado allí.
- **Qué esperar del parámetro.** `model_complexity=0` prioriza velocidad,
  `1` equilibra, y `2` prioriza precisión. Las diferencias se notan más en
  escenas complejas (oclusión, iluminación difícil, video con ruido). Con
  `static_image_mode=False` el tracking puede suavizar diferencias; con
  `static_image_mode=True` tienden a ser más visibles.

---

## NOTA:
En entornos Windows, si el vídeo con landmarks no se genera, reinstalar opencv-python puede resolver conflictos de codecs (OpenH264/FFmpeg).
uv pip uninstall opencv-python opencv-contrib-python opencv-python-headless uv pip install opencv-python
