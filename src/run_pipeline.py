"""Command-line runner for the analysis pipeline.

This module exposes a small CLI so the pipeline can be executed with
``python -m src.run_pipeline``. It mirrors the workflow used by the GUI
and Streamlit front-ends: process a video, count the repetitions and
persist the artefacts that the UI expects (metrics CSV and reps count
file).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

from src import config
from src.pipeline import run_full_pipeline_in_memory

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_COUNTS_DIR = DEFAULT_OUTPUT_DIR / "counts"
DEFAULT_POSES_DIR = DEFAULT_OUTPUT_DIR / "poses"


def _positive_int(value: str) -> int:
    """Return ``value`` as a positive integer or raise ``ArgumentTypeError``."""
    try:
        number = int(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise argparse.ArgumentTypeError(f"{value!r} no es un entero válido") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("El valor debe ser un entero positivo")
    return number


def _non_negative_float(value: str) -> float:
    """Return ``value`` as a non-negative float or raise ``ArgumentTypeError``."""
    try:
        number = float(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise argparse.ArgumentTypeError(f"{value!r} no es un número válido") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("El valor debe ser mayor o igual que 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser used by the CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline completo de análisis sobre un vídeo."
    )
    parser.add_argument("--video", required=True, help="Ruta al archivo de vídeo a procesar")
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta base donde guardar resultados (por defecto data/processed).",
    )
    parser.add_argument(
        "--sample_rate",
        type=_positive_int,
        default=config.DEFAULT_SAMPLE_RATE,
        help="Procesa 1 de cada N frames del vídeo.",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=None,
        help="Rotación manual en grados (0, 90, 180 o 270). Si se omite se auto-detecta.",
    )
    parser.add_argument(
        "--fps",
        type=_non_negative_float,
        default=None,
        help="FPS real del vídeo para corregir el cálculo de velocidades angulares.",
    )
    parser.add_argument(
        "--low_thresh",
        type=float,
        default=config.SQUAT_LOW_THRESH,
        help="Umbral bajo (en grados) usado para contar repeticiones.",
    )
    parser.add_argument(
        "--high_thresh",
        type=float,
        default=config.SQUAT_HIGH_THRESH,
        help="Umbral alto (no utilizado actualmente, reservado para futuros análisis).",
    )
    parser.add_argument(
        "--disable_crop",
        action="store_true",
        help="Desactiva el recorte automático del estimador de pose.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activa el modo depuración y guarda CSVs intermedios en la sesión.",
    )
    parser.add_argument(
        "--generate_debug_video",
        action="store_true",
        help="Genera un vídeo MP4 con el esqueleto renderizado.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra mensajes de log detallados durante la ejecución.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Initialise logging with a compact format."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(name)s: %(message)s")


def _ensure_directories(paths: Iterable[Path]) -> None:
    """Create the directories listed in ``paths`` if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point used by ``python -m src.run_pipeline``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    video_path = Path(args.video).expanduser()
    if not video_path.is_file():
        parser.error(f"No se encontró el vídeo: {video_path}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir_resolved = output_dir.resolve()
    default_output_resolved = DEFAULT_OUTPUT_DIR.resolve()

    if output_dir_resolved == default_output_resolved:
        counts_dir = DEFAULT_COUNTS_DIR
        poses_dir = DEFAULT_POSES_DIR
    else:
        counts_dir = output_dir / "counts"
        poses_dir = output_dir / "poses"
    _ensure_directories([output_dir, counts_dir, poses_dir])

    settings = {
        "output_dir": str(output_dir),
        "sample_rate": args.sample_rate,
        "rotate": args.rotate,
        "target_width": config.DEFAULT_TARGET_WIDTH,
        "target_height": config.DEFAULT_TARGET_HEIGHT,
        "use_crop": not args.disable_crop,
        "generate_debug_video": args.generate_debug_video,
        "debug_mode": args.debug,
        "low_thresh": args.low_thresh,
        "high_thresh": args.high_thresh,
        "fps_override": args.fps,
    }

    try:
        results = run_full_pipeline_in_memory(str(video_path), settings)
    except Exception as exc:  # pragma: no cover - surfaced to the CLI user
        LOGGER.exception("Fallo ejecutando el pipeline")
        return 1

    repetitions = results.get("repeticiones_contadas", 0)
    metrics_df = results.get("dataframe_metricas")

    video_name = video_path.stem
    count_path = counts_dir / f"{video_name}_count.txt"
    count_path.write_text(f"{repetitions}\n", encoding="utf-8")

    if metrics_df is not None:
        metrics_path = poses_dir / f"{video_name}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
    else:  # pragma: no cover - run_full_pipeline_in_memory always devuelve DF
        LOGGER.warning("No se recibió DataFrame de métricas; se omite la exportación")

    print(f"Repeticiones detectadas: {repetitions}")
    if metrics_df is not None:
        print(f"CSV de métricas: {metrics_path}")
    print(f"Archivo de conteo: {count_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
