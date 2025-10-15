# app.py
import sys
import tempfile
from pathlib import Path

import streamlit as st


# Ensure the project root is available on the import path when Streamlit executes the app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src import config
from src.pipeline import run_full_pipeline_in_memory
from src.run_pipeline import DEFAULT_COUNTS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_POSES_DIR

st.title("Gym Performance Analysis")

uploaded = st.file_uploader("Sube un vídeo de sentadilla", type=["mp4","mov"])
low = st.slider("Umbral bajo (°)",  0, 180, 80)
high = st.slider("Umbral alto (°)", 0, 180, 150)
fps = st.number_input("FPS del vídeo", 1.0, 120.0, 28.57)

if uploaded is not None:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mov")
    tmp_file.write(uploaded.read())
    tmp_file.flush()
    tmp_file.close()
    video_path = tmp_file.name
    st.video(video_path)

    if st.button("Empezar análisis"):
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_COUNTS_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_POSES_DIR.mkdir(parents=True, exist_ok=True)

        settings = {
            "output_dir": str(DEFAULT_OUTPUT_DIR),
            "sample_rate": 1,
            "rotate": None,
            "target_width": config.DEFAULT_TARGET_WIDTH,
            "target_height": config.DEFAULT_TARGET_HEIGHT,
            "use_crop": True,
            "generate_debug_video": False,
            "debug_mode": False,
            "low_thresh": low,
            "high_thresh": high,
            "fps_override": fps,
        }

        with st.spinner("Procesando vídeo…"):
            try:
                results = run_full_pipeline_in_memory(video_path, settings)
            except Exception as exc:  # pragma: no cover - surfaced to the UI user
                st.error("Ha ocurrido un error durante el análisis")
                st.code(str(exc))
                results = None

        if results is not None:
            repetitions = results.get("repeticiones_contadas", 0)
            metrics_df = results.get("dataframe_metricas")

            video_stem = Path(video_path).stem
            count_path = DEFAULT_COUNTS_DIR / f"{video_stem}_count.txt"
            count_path.write_text(f"{repetitions}\n", encoding="utf-8")

            st.success(f"Análisis completado ✅ | Repeticiones detectadas: {repetitions}")
            st.markdown("### Recuento de repeticiones")
            st.code(f"{repetitions}")

            if metrics_df is not None:
                metrics_path = DEFAULT_POSES_DIR / f"{video_stem}_metrics.csv"
                metrics_df.to_csv(metrics_path, index=False)
                st.markdown("### Métricas calculadas")
                st.dataframe(metrics_df)

            st.download_button(
                "Descargar recuento",
                data=count_path.read_text(encoding="utf-8"),
                file_name=f"{video_stem}_count.txt",
                mime="text/plain",
            )
