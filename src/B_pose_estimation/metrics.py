# src/B_pose_estimation/metrics.py
"""
Contiene funciones puras para el cálculo de métricas biomecánicas.
Responsabilidad: matemáticas y biomecánica.
"""
import numpy as np
import pandas as pd
import math

def normalize_landmarks(landmarks):
    """Centra los landmarks en el punto medio de la cadera."""
    hip_left, hip_right = landmarks[23], landmarks[24]
    cx = (hip_left['x'] + hip_right['x']) / 2.0
    cy = (hip_left['y'] + hip_right['y']) / 2.0
    return [{'x': lm['x'] - cx, 'y': lm['y'] - cy, 'z': lm['z'], 'visibility': lm['visibility']} for lm in landmarks]

def calculate_angle(p1, p2, p3):
    """Calcula el ángulo (en grados) formado por tres puntos."""
    v1, v2 = (p1['x'] - p2['x'], p1['y'] - p2['y']), (p3['x'] - p2['x'], p3['y'] - p2['y'])
    dot_product, mag1, mag2 = v1[0] * v2[0] + v1[1] * v2[1], math.hypot(v1[0], v1[1]), math.hypot(v2[0], v2[1])
    if mag1 * mag2 == 0: return 0.0
    return math.degrees(math.acos(max(min(dot_product / (mag1 * mag2), 1.0), -1.0)))

def extract_joint_angles(landmarks):
    """Extrae un diccionario de ángulos clave de las articulaciones."""
    return {
        'rodilla_izq': calculate_angle(landmarks[23], landmarks[25], landmarks[27]),
        'rodilla_der': calculate_angle(landmarks[24], landmarks[26], landmarks[28]),
        'codo_izq': calculate_angle(landmarks[11], landmarks[13], landmarks[15]),
        'codo_der': calculate_angle(landmarks[12], landmarks[14], landmarks[16]),
    }

def calculate_distances(landmarks):
    """Calcula distancias clave."""
    return {
        'anchura_hombros': abs(landmarks[12]['x'] - landmarks[11]['x']),
        'separacion_pies': abs(landmarks[28]['x'] - landmarks[27]['x']),
    }

def calculate_angular_velocity(angle_sequence, fps):
    """Calcula la velocidad angular de una secuencia de ángulos."""
    if not angle_sequence or fps == 0: return [0.0] * len(angle_sequence)
    velocities = [0.0]
    dt = 1.0 / fps
    for i in range(1, len(angle_sequence)):
        velocities.append(abs(angle_sequence[i] - angle_sequence[i - 1]) / dt)
    return velocities

def calculate_symmetry(angle_left, angle_right):
    """Calcula la simetría entre dos ángulos."""
    if pd.isna(angle_left) or pd.isna(angle_right): return np.nan
    max_angle = max(abs(angle_left), abs(angle_right))
    if max_angle == 0: return 1.0
    return 1.0 - (abs(angle_left - angle_right) / max_angle)