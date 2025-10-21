"""
Default tunable parameters and settings for the analysis pipeline and GUI.
"""

# --- PIPELINE PARAMETERS ---
MODEL_COMPLEXITY = 1
DEFAULT_TARGET_WIDTH = 256
DEFAULT_TARGET_HEIGHT = 256

# --- COUNTING PARAMETERS (legacy) ---
SQUAT_HIGH_THRESH = 160.0
SQUAT_LOW_THRESH = 100.0
PEAK_PROMINENCE = 10  # Prominence used by the peak detector
PEAK_DISTANCE = 15    # Minimum distance in frames between repetitions

# --- DEFAULT GUI / APP VALUES ---
DEFAULT_SAMPLE_RATE = 3
DEFAULT_ROTATION = "0"
DEFAULT_USE_CROP = True
DEFAULT_GENERATE_VIDEO = True
DEFAULT_DEBUG_MODE = True
DEFAULT_DARK_MODE = False
