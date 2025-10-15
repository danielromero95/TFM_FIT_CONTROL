"""Command-line runner for the analysis pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

from src import config
from src.pipeline import Report, run_pipeline

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_COUNTS_DIR = DEFAULT_OUTPUT_DIR / "counts"
DEFAULT_POSES_DIR = DEFAULT_OUTPUT_DIR / "poses"


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise argparse.ArgumentTypeError(f"{value!r} no es un entero válido") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("El valor debe ser un entero positivo")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise argparse.ArgumentTypeError(f"{value!r} no es un número válido") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("El valor debe ser mayor que 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline completo de análisis sobre un vídeo.",
    )
    parser.add_argument("--video", required=True, help="Ruta al archivo de vídeo a procesar")
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta base donde guardar resultados (por defecto data/processed).",
    )
    parser.add_argument(
        "--target_fps",
        type=_positive_float,
        default=None,
        help="FPS objetivo tras el muestreo. Si se omite se usa la configuración por defecto.",
    )
    parser.add_argument(
        "--sample_rate",
        type=_positive_int,
        default=None,
        help="Compatibilidad: procesa 1 de cada N frames (sobrescribe target_fps si se usa).",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=None,
        help="Rotación manual en grados (0, 90, 180 o 270). Si se omite se auto-detecta.",
    )
    parser.add_argument(
        "--low_thresh",
        type=float,
        default=config.SQUAT_LOW_THRESH,
        help="Umbral bajo (en grados) usado para la evaluación de profundidad.",
    )
    parser.add_argument(
        "--high_thresh",
        type=float,
        default=config.SQUAT_HIGH_THRESH,
        help="Umbral alto para la evaluación de profundidad.",
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
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(name)s: %(message)s")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    video_path = Path(args.video).expanduser()
    if not video_path.is_file():
        parser.error(f"No se encontró el vídeo: {video_path}")

    output_dir = Path(args.output_dir).expanduser()
    if args.sample_rate is not None and args.sample_rate <= 0:
        parser.error("sample_rate debe ser mayor que 0")

    cfg = config.load_default()
    cfg.output.base_dir = output_dir
    cfg.output.counts_dir = DEFAULT_COUNTS_DIR if output_dir == DEFAULT_OUTPUT_DIR else output_dir / "counts"
    cfg.output.poses_dir = DEFAULT_POSES_DIR if output_dir == DEFAULT_OUTPUT_DIR else output_dir / "poses"

    if args.target_fps is not None:
        cfg.video.target_fps = float(args.target_fps)
    if args.sample_rate is not None:
        cfg.video.manual_sample_rate = int(args.sample_rate)
    cfg.pose.rotate = args.rotate
    cfg.pose.use_crop = not args.disable_crop
    cfg.debug.generate_debug_video = args.generate_debug_video
    cfg.debug.debug_mode = args.debug
    cfg.faults.low_thresh = float(args.low_thresh)
    cfg.faults.high_thresh = float(args.high_thresh)

    try:
        report: Report = run_pipeline(str(video_path), cfg)
    except Exception as exc:  # pragma: no cover - surfaced to the CLI user
        LOGGER.exception("Fallo ejecutando el pipeline")
        return 1

    repetitions = report.repetitions
    if report.stats.skip_reason:
        LOGGER.warning("Conteo omitido: %s", report.stats.skip_reason)
        repetitions = 0

    metrics_df = report.metrics

    counts_dir = cfg.output.counts_dir
    poses_dir = cfg.output.poses_dir
    counts_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem
    count_path = counts_dir / f"{video_name}_count.txt"
    count_path.write_text(f"{repetitions}\n", encoding="utf-8")

    metrics_path: Optional[Path] = None
    if metrics_df is not None:
        metrics_path = poses_dir / f"{video_name}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
    else:  # pragma: no cover - run_pipeline siempre devuelve un DataFrame
        LOGGER.warning("No se recibió DataFrame de métricas; se omite la exportación")

    print(f"Repeticiones detectadas: {repetitions}")
    if metrics_path is not None:
        print(f"CSV de métricas: {metrics_path}")
    print(f"Archivo de conteo: {count_path}")
    print(f"CONFIG_SHA1: {report.stats.config_sha1}")
    if report.stats.skip_reason:
        print(f"AVISO: {report.stats.skip_reason}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
