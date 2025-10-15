# src/app.py
"""Streamlit front-end for the analysis pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Ensure Windows loads codec DLLs from the active conda env first -----------
if sys.platform.startswith("win"):
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        dll_dir = Path(conda_prefix) / "Library" / "bin"
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

# --- Tame noisy logs from TF/MediaPipe -----------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# Ensure the project root is available on the import path when Streamlit executes the app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.pipeline import Report, run_pipeline

st.title("Gym Performance Analysis")

uploaded = st.file_uploader(
    "Sube un vídeo de sentadilla",
    type=["mp4", "mov", "avi", "mkv", "mpg", "mpeg", "wmv"],
)
low = st.slider("Umbral bajo (°)", 0, 180, 80)
high = st.slider("Umbral alto (°)", 0, 180, 150)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".mp4"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uploaded.read())
    tmp_file.flush()
    tmp_file.close()
    video_path = Path(tmp_file.name)

    st.video(str(video_path))

    cfg = config.load_default()
    cfg.faults.low_thresh = float(low)
    cfg.faults.high_thresh = float(high)
    # Reduce MediaPipe warning noise; resizing still happens before pose estimation
    cfg.pose.use_crop = False

    st.markdown(f"**CONFIG_SHA1:** `{cfg.fingerprint()}`")

    if st.button("Empezar análisis"):
        with st.spinner("Procesando vídeo…"):
            try:
                report: Report = run_pipeline(str(video_path), cfg)
            except Exception as exc:  # pragma: no cover - surfaced to the UI user
                st.error("Ha ocurrido un error durante el análisis")
                st.code(str(exc))
                report = None

        if report is not None:
            stats = report.stats
            repetitions = report.repetitions
            metrics_df = report.metrics

            counts_dir = Path(cfg.output.counts_dir)
            poses_dir = Path(cfg.output.poses_dir)
            counts_dir.mkdir(parents=True, exist_ok=True)
            poses_dir.mkdir(parents=True, exist_ok=True)

            video_stem = video_path.stem
            count_path = counts_dir / f"{video_stem}_count.txt"
            count_path.write_text(f"{repetitions}\n", encoding="utf-8")

            if metrics_df is not None:
                metrics_path = poses_dir / f"{video_stem}_metrics.csv"
                metrics_df.to_csv(metrics_path, index=False)
            else:
                metrics_path = None

            st.success(
                "Análisis completado ✅" if stats.skip_reason is None else "Análisis completado con avisos ⚠️"
            )
            st.markdown(f"### Repeticiones detectadas: **{repetitions}**")

            st.markdown("### Estadísticas de la ejecución")
            stats_rows = [
                {"Campo": "CONFIG_SHA1", "Valor": stats.config_sha1},
                {"Campo": "fps_original", "Valor": f"{stats.fps_original:.2f}"},
                {"Campo": "fps_effective", "Valor": f"{stats.fps_effective:.2f}"},
                {"Campo": "frames", "Valor": stats.frames},
                {"Campo": "exercise_detected", "Valor": stats.exercise_detected},
                {"Campo": "primary_angle", "Valor": stats.primary_angle or "N/D"},
                {"Campo": "angle_range_deg", "Valor": f"{stats.angle_range_deg:.2f}"},
                {"Campo": "min_prominence", "Valor": f"{stats.min_prominence:.2f}"},
                {"Campo": "min_distance_sec", "Valor": f"{stats.min_distance_sec:.2f}"},
                {"Campo": "refractory_sec", "Valor": f"{stats.refractory_sec:.2f}"},
            ]
            stats_df = pd.DataFrame(stats_rows, columns=["Campo", "Valor"]).astype({"Valor": "string"})
            try:
                st.dataframe(stats_df, width="stretch")
            except Exception:
                st.json({row["Campo"]: row["Valor"] for row in stats_rows})

            if stats.config_path:
                st.info(f"Configuración utilizada guardada en: `{stats.config_path}`")

            if stats.warnings:
                st.warning("\n".join(f"• {msg}" for msg in stats.warnings))

            if stats.skip_reason:
                st.error(f"Conteo de repeticiones omitido: {stats.skip_reason}")

            st.markdown("### Recuento de repeticiones")
            st.code(f"{repetitions}")

            if metrics_df is not None:
                st.markdown("### Métricas calculadas")
                st.dataframe(metrics_df, use_container_width=True)

            if metrics_path is not None:
                st.download_button(
                    "Descargar métricas",
                    data=metrics_path.read_text(encoding="utf-8"),
                    file_name=f"{video_stem}_metrics.csv",
                    mime="text/csv",
                )

            st.download_button(
                "Descargar recuento",
                data=count_path.read_text(encoding="utf-8"),
                file_name=f"{video_stem}_count.txt",
                mime="text/plain",
            )

            if report.debug_video_path:
                st.markdown("### Vídeo de depuración")
                st.video(str(report.debug_video_path))
