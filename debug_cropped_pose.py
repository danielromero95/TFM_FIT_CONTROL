import cv2
import os
import numpy as np
from mediapipe.python.solutions import pose as mp_pose_module
from mediapipe.python.solutions import drawing_utils as mp_drawing_utils

# ------------ Parámetros de prueba ---------------
IMG_PATH = "data/processed/images/1-Squat_Own/frame_0020.jpg"
CROP_MARGIN = 0.15    # el mismo valor que pusiste en CroppedPoseEstimator
TARGET_SIZE = (256, 256)

# 1) Clase CroppedPoseEstimator (cópiala exactamente de tu pose_utils.py, incluyendo imports)
class CroppedPoseEstimator:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 crop_margin=0.15,
                 target_size=(256,256)):
        self.crop_margin = crop_margin
        self.target_size = target_size  # (width, height) del recorte final

        # Importamos mediapipe aquí para no romper el env si nunca se llama
        from mediapipe.python.solutions import pose as mp_pose_module
        from mediapipe.python.solutions import drawing_utils as mp_drawing_utils
        self.mp_pose_module = mp_pose_module
        self.mp_drawing_utils = mp_drawing_utils

        # Creamos un único objeto de Mediapipe Pose para la detección en full y en crop
        self.pose_full = self.mp_pose_module.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )
        self.pose_crop = self.mp_pose_module.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )

    def estimate_and_crop(self, image_bgr):
        h0, w0 = image_bgr.shape[:2]
        # ---------------------------------------------------------------
        #  PASO 1: estimar pose en toda la imagen
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results_full = self.pose_full.process(img_rgb)
        if not results_full.pose_landmarks:
            return None, image_bgr  # no detectó

        # ---------------------------------------------------------------
        #  PASO 2: convierto cada landmark normalizado en píxeles
        lms = results_full.pose_landmarks.landmark  # lista de 33 landmarks
        xy = np.array([[lm.x * w0, lm.y * h0] for lm in lms])  # (33,2) array

        # ---------------------------------------------------------------
        #  PASO 3: construyo bounding-box minimal que encierra todos los puntos
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        dx = (x_max - x_min) * self.crop_margin
        dy = (y_max - y_min) * self.crop_margin
        x1 = max(int(x_min - dx), 0)
        y1 = max(int(y_min - dy), 0)
        x2 = min(int(x_max + dx), w0-1)
        y2 = min(int(y_max + dy), h0-1)

        # ---------------------------------------------------------------
        #  PASO 4: recorto la región en torno a la persona
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, image_bgr

        # ---------------------------------------------------------------
        #  PASO 5: redimensiono el recorte y vuelvo a estimar pose
        w_tgt, h_tgt = self.target_size
        crop_resized = cv2.resize(crop, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        results_crop = self.pose_crop.process(crop_rgb)

        # ---------------------------------------------------------------
        #  PASO 6: recorremos los landmarks detectados en el recorte
        landmarks_crop = []
        annotated_crop = crop_resized.copy()
        if results_crop.pose_landmarks:
            for lm in results_crop.pose_landmarks.landmark:
                landmarks_crop.append({
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })
            # Dibujar landmarks en la copia del crop
            self.mp_drawing_utils.draw_landmarks(
                annotated_crop,
                results_crop.pose_landmarks,
                self.mp_pose_module.POSE_CONNECTIONS
            )
        else:
            landmarks_crop = None

        return landmarks_crop, annotated_crop

    def close(self):
        self.pose_full.close()
        self.pose_crop.close()


# ============================================================
#  Script de prueba:
# ============================================================
if __name__ == "__main__":
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"[ERROR] No se pudo leer {IMG_PATH}")
        exit(1)

    # 1) Primera pasada
    estimator = CroppedPoseEstimator(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        crop_margin=CROP_MARGIN,
        target_size=TARGET_SIZE
    )

    # 2) Obtenemos los landmarks y el crop anotado
    landmarks_crop, annotated_crop = estimator.estimate_and_crop(img)
    estimator.close()

    # 3) Guardamos ambas imágenes (full + crop) para verlas:
    os.makedirs("debug_outputs", exist_ok=True)
    cv2.imwrite("debug_outputs/debug_full_for_crop.jpg", img)          # imagen original
    if landmarks_crop is None:
        print("[DEBUG] CroppedPoseEstimator NO detectó nada en el crop final.")
        cv2.imwrite("debug_outputs/debug_crop_empty.jpg", annotated_crop)
    else:
        print(f"[DEBUG] CroppedPoseEstimator sí detectó {len(landmarks_crop)} landmarks en el crop final.")
        cv2.imwrite("debug_outputs/debug_crop_annotated.jpg", annotated_crop)
