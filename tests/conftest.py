# tests/conftest.py
"""Test configuration helpers."""
from pathlib import Path
import sys

# Repo root = parent de 'tests'
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Asegura que los paquetes en src/ son importables
src_str = str(SRC)
if src_str not in sys.path:
    sys.path.insert(0, src_str)

# (Opcional) si quieres mantener también la raíz, ponla detrás
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.append(root_str)
