# src/app.py
"""Streamlit front-end for the analysis pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

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
from src.detect.exercise_detector import detect_exercise
from src.pipeline import Report, run_pipeline

EXERCISE_OPTIONS = ["Auto (MVP)", "Sentadilla"]
EXERCISE_TO_CONFIG = {
    "Auto (MVP)": "auto",
    "Sentadilla": "squat",
}
CONFIG_DEFAULTS: Dict[str, float | str | bool | None] = {
    "low": 80,
    "high": 150,
    "primary_angle": "rodilla_izq",
    "min_prominence": 10.0,
    "min_distance_sec": 0.5,
    "debug_video": True,
    "use_crop": True,
    "target_fps": None,
}


def _init_session_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = "upload"
    if "upload_data" not in st.session_state:
        st.session_state.upload_data = None
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "exercise" not in st.session_state:
        st.session_state.exercise = EXERCISE_OPTIONS[0]
    if "detect_result" not in st.session_state:
        st.session_state.detect_result = None
    if "configure_values" not in st.session_state:
        st.session_state.configure_values = CONFIG_DEFAULTS.copy()
    if "report" not in st.session_state:
        st.session_state.report = None
    if "pipeline_error" not in st.session_state:
        st.session_state.pipeline_error = None
    if "count_path" not in st.session_state:
        st.session_state.count_path = None
    if "metrics_path" not in st.session_state:
        st.session_state.metrics_path = None
    if "cfg_fingerprint" not in st.session_state:
        st.session_state.cfg_fingerprint = None


def _reset_app() -> None:
    video_path = st.session_state.get("video_path")
    if video_path:
        try:
            Path(video_path).unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            # Python < 3.8 compatibility – ignore missing files silently
            try:
                Path(video_path).unlink()
            except FileNotFoundError:
                pass
        except OSError:
            pass
    for key in [
        "video_path",
        "report",
        "pipeline_error",
        "count_path",
        "metrics_path",
        "cfg_fingerprint",
    ]:
        st.session_state.pop(key, None)
    if "upload_data" in st.session_state:
        st.session_state.upload_data = None
    st.session_state.exercise = EXERCISE_OPTIONS[0]
    st.session_state.detect_result = None
    st.session_state.configure_values = CONFIG_DEFAULTS.copy()
    st.session_state.step = "upload"
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def _ensure_video_path() -> None:
    upload_data = st.session_state.upload_data
    if not upload_data:
        return
    # If we're replacing an existing temp file, delete it first
    old_path = st.session_state.get("video_path")
    if old_path:
        try:
            Path(old_path).unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            try:
                Path(old_path).unlink()
            except FileNotFoundError:
                pass
        except OSError:
            pass
    suffix = Path(upload_data["name"]).suffix or ".mp4"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(upload_data["bytes"])
    tmp_file.flush()
    tmp_file.close()
    st.session_state.video_path = tmp_file.name
    st.session_state.detect_result = None
    # Free large byte payload from session to reduce memory usage once persisted
    st.session_state.upload_data = None


def _run_pipeline(progress_cb=None) -> None:
    st.session_state.pipeline_error = None
    st.session_state.report = None
    st.session_state.count_path = None
    st.session_state.metrics_path = None
    st.session_state.cfg_fingerprint = None

    video_path = st.session_state.get("video_path")
    if not video_path:
        st.session_state.pipeline_error = "No se encontró el vídeo a procesar."
        return

    cfg = config.load_default()

    cfg_values = st.session_state.get("configure_values", CONFIG_DEFAULTS)
    cfg.faults.low_thresh = float(cfg_values.get("low", CONFIG_DEFAULTS["low"]))
    cfg.faults.high_thresh = float(cfg_values.get("high", CONFIG_DEFAULTS["high"]))
    cfg.counting.primary_angle = str(cfg_values.get("primary_angle", CONFIG_DEFAULTS["primary_angle"]))
    cfg.counting.min_prominence = float(cfg_values.get("min_prominence", CONFIG_DEFAULTS["min_prominence"]))
    cfg.counting.min_distance_sec = float(cfg_values.get("min_distance_sec", CONFIG_DEFAULTS["min_distance_sec"]))
    cfg.debug.generate_debug_video = bool(cfg_values.get("debug_video", True))
    cfg.pose.use_crop = bool(cfg_values.get("use_crop", True))
    tfps = cfg_values.get("target_fps")
    if tfps not in (None, "", 0) and hasattr(cfg, "video"):
        try:
            cfg.video.target_fps = float(tfps)
            if hasattr(cfg.video, "manual_sample_rate"):
                cfg.video.manual_sample_rate = None
        except (TypeError, ValueError):
            pass

    exercise_label = st.session_state.get("exercise", EXERCISE_OPTIONS[0])
    cfg.counting.exercise = EXERCISE_TO_CONFIG.get(exercise_label, "squat")

    det = st.session_state.get("detect_result")
    prefetched_detection = None
    if det:
        prefetched_detection = (
            det.get("label", "unknown"),
            det.get("view", "unknown"),
            float(det.get("confidence", 0.0)),
        )

    try:
        report: Report = run_pipeline(
            str(video_path),
            cfg,
            progress_callback=progress_cb,
            prefetched_detection=prefetched_detection,
        )
    except Exception as exc:  # pragma: no cover - surfaced to the UI user
        st.session_state.pipeline_error = str(exc)
        return

    st.session_state.report = report
    st.session_state.cfg_fingerprint = report.stats.config_sha1

    counts_dir = Path(cfg.output.counts_dir)
    poses_dir = Path(cfg.output.poses_dir)
    counts_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    count_path = counts_dir / f"{video_stem}_count.txt"
    count_path.write_text(f"{report.repetitions}\n", encoding="utf-8")
    st.session_state.count_path = str(count_path)

    metrics_df = report.metrics
    if metrics_df is not None:
        metrics_path = poses_dir / f"{video_stem}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        st.session_state.metrics_path = str(metrics_path)
    else:
        st.session_state.metrics_path = None


def _upload_step() -> None:
    st.markdown("### 1. Sube tu vídeo")
    uploaded = st.file_uploader(
        "Sube un vídeo",
        type=["mp4", "mov", "avi", "mkv", "mpg", "mpeg", "wmv"],
    )
    if uploaded is not None:
        st.session_state.upload_data = {
            "name": uploaded.name,
            "bytes": uploaded.getvalue(),
        }
    can_continue = st.session_state.upload_data is not None
    if st.button("Continuar", disabled=not can_continue):
        if can_continue:
            _ensure_video_path()
            st.session_state.step = "detect"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()


def _detect_step() -> None:
    st.markdown("### 2. Detecta el ejercicio")
    if st.session_state.video_path:
        st.video(str(st.session_state.video_path))

    detect_disabled = st.session_state.video_path is None
    if st.button("Detectar ejercicio (beta)", disabled=detect_disabled):
        video_path = st.session_state.get("video_path")
        if not video_path:
            st.warning("Sube un vídeo antes de detectar el ejercicio.")
        else:
            with st.spinner("Analizando el vídeo para detectar el ejercicio…"):
                label, view, confidence = detect_exercise(str(video_path))
            if label == "unknown" and view == "unknown" and confidence <= 0.0:
                st.session_state.detect_result = None
                st.warning(
                    "No se pudo determinar el ejercicio automáticamente. "
                    "Revisa la grabación o selecciona el ejercicio manualmente."
                )
            else:
                st.session_state.detect_result = {
                    "label": label,
                    "view": view,
                    "confidence": float(confidence),
                }
                if label == "squat":
                    st.session_state.exercise = "Sentadilla"
                else:
                    st.session_state.exercise = EXERCISE_OPTIONS[0]

    detect_result = st.session_state.get("detect_result")
    if detect_result:
        label = detect_result["label"]
        view = detect_result["view"]
        confidence_pct = int(round(detect_result["confidence"] * 100))
        st.info(
            f"Ejercicio detectado: {label} · Vista: {view} · "
            f"Confianza: {confidence_pct}%"
        )

    st.selectbox(
        "Selecciona el ejercicio",
        options=EXERCISE_OPTIONS,
        key="exercise",
    )

    col_back, col_forward = st.columns(2)
    with col_back:
        if st.button("Atrás", key="detect_back"):
            st.session_state.step = "upload"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with col_forward:
        if st.button("Continuar", key="detect_continue"):
            st.session_state.step = "configure"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()


def _configure_step() -> None:
    st.markdown("### 3. Configura el análisis")
    cfg_values = st.session_state.get("configure_values", CONFIG_DEFAULTS.copy())

    col1, col2 = st.columns(2)
    with col1:
        low = st.number_input(
            "Umbral bajo (°)",
            min_value=0,
            max_value=180,
            value=int(cfg_values.get("low", CONFIG_DEFAULTS["low"])),
        )
    with col2:
        high = st.number_input(
            "Umbral alto (°)",
            min_value=0,
            max_value=180,
            value=int(cfg_values.get("high", CONFIG_DEFAULTS["high"])),
        )

    primary_angle = st.text_input(
        "Ángulo primario",
        value=str(cfg_values.get("primary_angle", CONFIG_DEFAULTS["primary_angle"])),
    )

    col3, col4 = st.columns(2)
    with col3:
        min_prominence = st.number_input(
            "Prominencia mínima",
            min_value=0.0,
            value=float(cfg_values.get("min_prominence", CONFIG_DEFAULTS["min_prominence"])),
            step=0.5,
        )
    with col4:
        min_distance_sec = st.number_input(
            "Distancia mínima (s)",
            min_value=0.0,
            value=float(cfg_values.get("min_distance_sec", CONFIG_DEFAULTS["min_distance_sec"])),
            step=0.1,
        )

    debug_video = st.checkbox(
        "Generar vídeo de depuración",
        value=bool(cfg_values.get("debug_video", CONFIG_DEFAULTS["debug_video"])),
    )
    use_crop = st.checkbox(
        "Usar recorte automático (MediaPipe)",
        value=bool(cfg_values.get("use_crop", CONFIG_DEFAULTS["use_crop"])),
    )

    target_fps_current = cfg_values.get("target_fps", CONFIG_DEFAULTS.get("target_fps"))
    target_fps_default = "" if target_fps_current in (None, "") else str(target_fps_current)
    target_fps_raw = st.text_input(
        "FPS objetivo tras muestreo",
        value=target_fps_default,
        help="Déjalo vacío para desactivar el remuestreo."
    )
    target_fps_error = False
    if target_fps_raw.strip():
        try:
            parsed_target_fps = float(target_fps_raw)
        except ValueError:
            target_fps_error = True
            target_fps_value = cfg_values.get("target_fps", None)
            st.warning("Introduce un número válido para FPS objetivo o deja el campo vacío.")
        else:
            target_fps_value = parsed_target_fps
    else:
        target_fps_value = None

    current_values = {
        "low": float(low),
        "high": float(high),
        "primary_angle": primary_angle,
        "min_prominence": float(min_prominence),
        "min_distance_sec": float(min_distance_sec),
        "debug_video": bool(debug_video),
        "use_crop": bool(use_crop),
        "target_fps": target_fps_value,
    }
    if not target_fps_error:
        st.session_state.configure_values = current_values

    col_back, col_forward = st.columns(2)
    with col_back:
        if st.button("Atrás", key="configure_back"):
            st.session_state.step = "detect"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with col_forward:
        if st.button("Analizar"):
            if target_fps_error:
                st.warning("Corrige el valor de FPS objetivo antes de continuar.")
            else:
                st.session_state.configure_values = current_values
                st.session_state.step = "running"
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()


def _running_step() -> None:
    st.markdown("### 4. Analizando…")
    progress_placeholder = st.progress(0)
    phase_placeholder = st.empty()
    debug_enabled = bool(
        st.session_state.get("configure_values", {}).get("debug_video", True)
    )

    def _phase_for(p: int) -> str:
        if p < 10:
            return "Preparando…"
        if p < 25:
            return "Extrayendo fotogramas…"
        if p < 50:
            return "Estimando pose…"
        if p < 65:
            return "Filtrando e interpolando…"
        if debug_enabled and p < 75:
            return "Renderizando vídeo de depuración…"
        if debug_enabled:
            if p < 90:
                return "Calculando métricas…"
        else:
            if p < 85:
                return "Calculando métricas…"
        if p < 100:
            return "Contando repeticiones…"
        return "Finalizando…"

    def _cb(p: int) -> None:
        p = max(0, min(100, int(p)))
        progress_placeholder.progress(p)
        phase_placeholder.text(_phase_for(p))

    phase_placeholder.text(_phase_for(0))

    _ensure_video_path()
    if not st.session_state.video_path:
        st.session_state.pipeline_error = "No se encontró el vídeo a procesar."
        st.session_state.step = "results"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
        return

    with st.spinner("Procesando vídeo…"):
        _run_pipeline(progress_cb=_cb)

    st.session_state.step = "results"
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def _results_step() -> None:
    st.markdown("### Resultados")
    if st.session_state.pipeline_error:
        st.error("Ha ocurrido un error durante el análisis")
        st.code(str(st.session_state.pipeline_error))
    elif st.session_state.report is not None:
        report: Report = st.session_state.report
        stats = report.stats
        repetitions = report.repetitions
        metrics_df = report.metrics

        st.success(
            "Análisis completado ✅" if stats.skip_reason is None else "Análisis completado con avisos ⚠️"
        )
        st.markdown(f"**CONFIG_SHA1:** `{st.session_state.cfg_fingerprint}`")
        st.markdown(f"### Repeticiones detectadas: **{repetitions}**")

        st.markdown("### Estadísticas de la ejecución")
        stats_rows = [
            {"Campo": "CONFIG_SHA1", "Valor": stats.config_sha1},
            {"Campo": "fps_original", "Valor": f"{stats.fps_original:.2f}"},
            {"Campo": "fps_effective", "Valor": f"{stats.fps_effective:.2f}"},
            {"Campo": "frames", "Valor": stats.frames},
            {
                "Campo": "exercise_selected",
                "Valor": stats.exercise_selected or "N/D",
            },
            {"Campo": "exercise_detected", "Valor": stats.exercise_detected},
            {"Campo": "view_detected", "Valor": stats.view_detected},
            {
                "Campo": "detection_confidence",
                "Valor": f"{stats.detection_confidence:.0%}",
            },
            {"Campo": "primary_angle", "Valor": stats.primary_angle or "N/D"},
            {"Campo": "angle_range_deg", "Valor": f"{stats.angle_range_deg:.2f}"},
            {"Campo": "min_prominence", "Valor": f"{stats.min_prominence:.2f}"},
            {"Campo": "min_distance_sec", "Valor": f"{stats.min_distance_sec:.2f}"},
            {"Campo": "refractory_sec", "Valor": f"{stats.refractory_sec:.2f}"},
        ]
        stats_df = pd.DataFrame(stats_rows, columns=["Campo", "Valor"]).astype({"Valor": "string"})
        try:
            st.dataframe(stats_df, use_container_width=True)
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
            numeric_columns = [
                col
                for col in metrics_df.columns
                if metrics_df[col].dtype.kind in "fi"
            ]
            if numeric_columns:
                default_selection = numeric_columns[:3]
                selected_metrics = st.multiselect(
                    "Visualizar métricas",
                    options=numeric_columns,
                    default=default_selection,
                )
                if selected_metrics:
                    st.line_chart(metrics_df[selected_metrics])

        if st.session_state.metrics_path is not None:
            metrics_data = None
            try:
                metrics_data = Path(st.session_state.metrics_path).read_text(
                    encoding="utf-8"
                )
            except FileNotFoundError:
                st.error("No se encontró el archivo de métricas para la descarga.")
            except OSError as exc:
                st.error(f"No se pudieron leer las métricas: {exc}")
            else:
                st.download_button(
                    "Descargar métricas",
                    data=metrics_data,
                    file_name=f"{Path(st.session_state.video_path).stem}_metrics.csv",
                    mime="text/csv",
                )

        if st.session_state.count_path is not None:
            count_data = None
            try:
                count_data = Path(st.session_state.count_path).read_text(
                    encoding="utf-8"
                )
            except FileNotFoundError:
                st.error("No se encontró el archivo de recuento para la descarga.")
            except OSError as exc:
                st.error(f"No se pudo leer el archivo de recuento: {exc}")
            else:
                st.download_button(
                    "Descargar recuento",
                    data=count_data,
                    file_name=f"{Path(st.session_state.video_path).stem}_count.txt",
                    mime="text/plain",
                )

        if report.debug_video_path and bool(
            st.session_state.get("configure_values", {}).get("debug_video", True)
        ):
            st.markdown("### Vídeo de depuración")
            st.video(str(report.debug_video_path))
    else:
        st.info("No se encontraron resultados para mostrar.")

    if st.session_state.video_path:
        st.markdown("### Vídeo original")
        st.video(str(st.session_state.video_path))

    # Fast path to adjust thresholds and re-run with the same video
    if st.button("Ajustar configuración y reanalizar"):
        st.session_state.step = "configure"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if st.button("Volver a inicio"):
        _reset_app()


st.title("Gym Performance Analysis")
_init_session_state()

step = st.session_state.step
if step == "upload":
    _upload_step()
elif step == "detect":
    _detect_step()
elif step == "configure":
    _configure_step()
elif step == "running":
    _running_step()
else:
    _results_step()
