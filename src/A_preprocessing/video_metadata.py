# src/A_preprocessing/video_metadata.py
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def _normalize_rotation(value: int) -> int:
    """Devuelve 0/90/180/270 a partir de un entero (tolera negativos)."""
    value = int(value) % 360
    # Redondeo al múltiplo de 90 más cercano
    candidates = [0, 90, 180, 270]
    return min(candidates, key=lambda x: abs(x - value))

def get_video_rotation(video_path: str) -> int:
    """
    Obtiene la rotación de los metadatos de vídeo usando ffprobe (si está disponible).
    Retorna 0 si no hay metadatos o ffprobe no está instalado.
    """
    try:
        if shutil.which("ffprobe") is None:
            logger.info("ffprobe no encontrado en PATH; se asume rotación 0.")
            return 0

        # 1) Intenta leer etiqueta clásica 'rotate'
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "default=nw=1:nk=1",
            str(Path(video_path))
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        txt = (res.stdout or "").strip()
        if txt:
            rot = _normalize_rotation(int(float(txt)))
            if rot:
                logger.info(f"Rotación detectada por ffprobe (tags): {rot}°")
            return rot

        # 2) (Opcional) Otros contenedores: si no hay 'rotate', devolvemos 0
        logger.info("No se detectó rotación en metadatos; se asume 0°.")
        return 0

    except Exception as exc:  # robustez
        logger.warning(f"No se pudo leer rotación con ffprobe ({exc}); se asume 0°.")
        return 0
