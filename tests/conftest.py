"""Test configuration helpers."""

import os
import sys

# Locate the project root (parent directory of ``tests``).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Ensure the project root is on PYTHONPATH for imports.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
