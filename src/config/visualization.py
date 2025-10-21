"""
Configuration for visualization, including MediaPipe landmarks and colors.
"""

# --- VISUALISATION CONFIGURATION ---
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (12, 14),
    (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30),
    (28, 32), (29, 31), (30, 32)
]
LANDMARK_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (0, 0, 255)  # Red
