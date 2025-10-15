# app.py
import streamlit as st
import tempfile, os
from subprocess import run

st.title("Gym Performance Analysis")

uploaded = st.file_uploader("Sube un vídeo de sentadilla", type=["mp4","mov"])
low = st.slider("Umbral bajo (°)",  0, 180, 80)
high = st.slider("Umbral alto (°)", 0, 180, 150)
fps = st.number_input("FPS del vídeo", 1.0, 120.0, 28.57)

if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mov")
    tfile.write(uploaded.read())
    video_path = tfile.name
    st.video(video_path)

    if st.button("Empezar análisis"):
        # Creamos un directorio de salida temporal
        outdir = tempfile.mkdtemp(prefix="gym_out_")
        # Llamamos al pipeline (igual que hicimos en consola)
        cmd = [
          "python", "-m", "src.run_pipeline",
          "--video", video_path,
          "--fps", str(fps),
          "--sample_rate", "1",
          "--low_thresh", str(low),
          "--high_thresh", str(high)
        ]
        with st.spinner("Procesando vídeo…"):
            res = run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            st.success("Análisis completado ✅")
            st.text(res.stdout)
            # Mostrar el TXT de repeticiones
            count_txt = os.path.join("data/processed/counts", os.path.basename(video_path).split(".")[0] + "_count.txt")
            if os.path.exists(count_txt):
                st.markdown("### Recuento de repeticiones:")
                st.code(open(count_txt).read())
        else:
            st.error("Ha ocurrido un error:")
            st.code(res.stderr)
