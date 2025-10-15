# tests/conftest.py

import sys, os

# Obtener la ruta al directorio padre de 'tests', que es la raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Añadir la carpeta raíz al PYTHONPATH
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
