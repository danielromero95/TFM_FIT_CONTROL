"""Parámetros por defecto y utilidades de configuración para el pipeline y la GUI."""

from __future__ import annotations

import os

from .constants import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE


def configure_environment() -> None:
    """Ajusta variables de entorno y rutas necesarias antes de lanzar la aplicación."""

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["GLOG_minloglevel"] = "2"

    try:
        from absl import logging as absl_logging  # type: ignore[import-not-found]
    except Exception:
        pass
    else:
        # Forzamos a ``absl`` a emitir solo errores para no saturar la consola.
        absl_logging.set_verbosity(absl_logging.ERROR)

    # El soporte Conda se eliminó por considerarse legado y ya no se ajusta el PATH.


# --- PARÁMETROS DEL PIPELINE ---
# Complejidad del grafo de MediaPipe (0/1/2). Usamos el modelo "heavy" (2)
# para priorizar precisión en vídeos complejos, asumiendo el coste extra.
MODEL_COMPLEXITY = 2

# Mantiene activado el uso del modelo base de pose sin segmentación de fondo,
# ya que la aplicación solo necesita *landmarks* corporales y no máscaras.
POSE_ENABLE_SEGMENTATION = False

# Desactiva el suavizado de la máscara de segmentación porque no usamos
# segmentación: evitamos trabajo innecesario si algún caller la activa.
POSE_SMOOTH_SEGMENTATION = False

# Mantiene un suavizado temporal de landmarks para reducir saltos entre frames
# y obtener ángulos más estables al contar repeticiones.
POSE_SMOOTH_LANDMARKS = True

# Indica que las entradas son vídeo (False): habilita el seguimiento entre
# frames y evita recalcular detecciones completas en cada paso.
POSE_STATIC_IMAGE_MODE = False

# Resolución objetivo a la que se reescala la entrada antes de inferir pose,
# balanceando coste computacional y nivel de detalle de los landmarks.
DEFAULT_TARGET_WIDTH = 256
DEFAULT_TARGET_HEIGHT = 256

# Frecuencia de refresco del panel de previsualización de la UI para no
# saturar recursos mientras se procesa el vídeo.
DEFAULT_PREVIEW_FPS = 10.0

# Tasas de muestreo que guían el procesado: valor preferido (por defecto el
# de la UI) y alternativas cuando el vídeo original es muy ligero o pesado.
POSE_TARGET_FPS_DEFAULT = DEFAULT_PREVIEW_FPS
POSE_TARGET_FPS_FALLBACK = 20.0
POSE_TARGET_SIZE_FALLBACK = 384
POSE_QUALITY_FALLBACK = 0.55

# Ritmo máximo al que se leen frames para detección cuando se prioriza tiempo
# real en streaming.
DETECTION_SAMPLE_FPS = 4.0

# Visibilidad mínima de un landmark para considerarlo válido en cálculos
# posteriores (conteo, métricas, overlay).
DEFAULT_LANDMARK_MIN_VISIBILITY = 0.5

# --- PARÁMETROS DE CONTEO ---
# Umbrales usados por el contador de repeticiones y la validación de profundidad.
# Rango de ángulos aceptado para clasificar una sentadilla como profunda.
SQUAT_HIGH_THRESH = 160.0
SQUAT_LOW_THRESH = 100.0

# Altura mínima de un pico (en grados) para que el detector de repeticiones lo
# considere una transición válida.
PEAK_PROMINENCE = 10

# --- VALORES PREDETERMINADOS PARA LA UI ---
# Activa el recorte centrado en la persona detectada para mejorar precisión
# de pose y evitar píxeles innecesarios.
DEFAULT_USE_CROP = True

# Genera un vídeo de depuración con overlays para inspeccionar resultados.
DEFAULT_GENERATE_VIDEO = True

# Permite mostrar diagnósticos adicionales en consola/GUI al depurar.
DEFAULT_DEBUG_MODE = True

# Periodo inicial en segundos antes de iniciar conteo para estabilizar la
# cámara y las detecciones.
DEFAULT_WARMUP_SECONDS = 0.5

# --- PROTECCIONES PARA MEDIOS PESADOS ---
# Ajustes que previenen bloqueos cuando se cargan vídeos/fotos muy grandes.
# Máximo tamaño del lado más largo para overlays generados (p. ej. 720p).
OVERLAY_MAX_LONG_SIDE = 1280

# Desactiva overlays cuando el artefacto resultante superaría este peso (40MB)
# evitando consumos excesivos de memoria y disco.
OVERLAY_DISABLE_OVER_BYTES = 40 * 1024 * 1024

# Deshabilita la previsualización en vivo cuando la imagen supera este límite
# de megapíxeles para no saturar la GPU/CPU.
PREVIEW_DISABLE_OVER_MP = 2.5

# Reduce los FPS de previsualización cuando el medio es pesado para priorizar
# estabilidad del procesamiento.
PREVIEW_MAX_FPS_HEAVY = 5.0

# Límite superior de FPS que se permite renderizar en el overlay para mantener
# un uso de recursos predecible en máquinas modestas.
OVERLAY_FPS_CAP = 15.0


def build_pose_kwargs(
    *,
    static_image_mode: bool | None = None,
    model_complexity: int | None = None,
    min_detection_confidence: float | None = None,
    min_tracking_confidence: float | None = None,
    smooth_landmarks: bool | None = None,
    enable_segmentation: bool | None = None,
) -> dict[str, object]:
    """Configuración estándar para el grafo ``Pose`` de MediaPipe.

    Centralizar estos parámetros garantiza que todos los puntos del pipeline
    utilicen el mismo modelo "heavy" (``model_complexity=2``) y suavizado de
    *landmarks*, priorizando la calidad sobre la velocidad para vídeos
    desafiantes.
    """

    return {
        "static_image_mode": POSE_STATIC_IMAGE_MODE if static_image_mode is None else static_image_mode,
        "model_complexity": MODEL_COMPLEXITY if model_complexity is None else model_complexity,
        "smooth_landmarks": POSE_SMOOTH_LANDMARKS if smooth_landmarks is None else bool(smooth_landmarks),
        "enable_segmentation": POSE_ENABLE_SEGMENTATION
        if enable_segmentation is None
        else bool(enable_segmentation),
        "smooth_segmentation": POSE_SMOOTH_SEGMENTATION,
        "min_detection_confidence": (
            MIN_DETECTION_CONFIDENCE if min_detection_confidence is None else float(min_detection_confidence)
        ),
        "min_tracking_confidence": (
            MIN_TRACKING_CONFIDENCE if min_tracking_confidence is None else float(min_tracking_confidence)
        ),
    }
