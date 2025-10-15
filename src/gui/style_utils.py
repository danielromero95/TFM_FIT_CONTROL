# src/gui/style_utils.py

import os
from PyQt5.QtWidgets import QApplication

def load_stylesheet(app: QApplication, project_root: str, dark: bool):
    """Carga la hoja de estilos (clara u oscura) para la aplicación."""
    themes_dir = os.path.join(project_root, 'themes')
    qss_file = 'dark.qss' if dark else 'light.qss'
    path = os.path.join(themes_dir, qss_file)
    
    if os.path.exists(path):
        with open(path, 'r') as f:
            app.setStyleSheet(f.read())
    else:
        print(f"Advertencia: No se encontró la hoja de estilos en {path}")