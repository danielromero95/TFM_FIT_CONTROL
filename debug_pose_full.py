import cv2
import os
from mediapipe.python.solutions import pose as mp_pose_module
from mediapipe.python.solutions import drawing_utils as mp_drawing_utils

# -------------- Cambia este path si tu frame 20 está en otro lugar ------------
IMG_PATH = "data/processed/images/1-Squat_Own/frame_0020.jpg"

# 1) Carga la imagen
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"[ERROR] No se pudo leer la imagen en {IMG_PATH}")
    exit(1)

# 2) Crea el objeto Pose de MediaPipe
pose = mp_pose_module.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# 3) Convierte a RGB y procesa
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(img_rgb)
pose.close()

# 4) Si no hay detección, avisamos y terminamos
if not results.pose_landmarks:
    print("[DEBUG] MediaPipe NO detectó pose en el frame 20 (sin crop).")
    cv2.imwrite("debug_full_no_pose.jpg", img)
    exit(0)

# 5) Si sí hay detección, dibuja sobre la copia y guarda
annotated = img.copy()
mp_drawing_utils.draw_landmarks(annotated, results.pose_landmarks, mp_pose_module.POSE_CONNECTIONS)
os.makedirs("debug_outputs", exist_ok=True)
cv2.imwrite("debug_outputs/debug_full_annotated.jpg", annotated)
print("[DEBUG] MediaPipe SÍ detectó pose. Imagen anotada guardada en debug_outputs/debug_full_annotated.jpg")
