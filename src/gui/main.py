# src/gui/main.py

import sys
import os
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings

# --- Importamos desde la nueva utilidad ---
from .style_utils import load_stylesheet

def find_project_root():
    path = os.path.abspath(os.path.dirname(__file__))
    while os.path.basename(path) != 'gym-performance-analysis':
        parent_path = os.path.dirname(path)
        if parent_path == path: return os.getcwd()
        path = parent_path
    return path

def setup_logging(project_root):
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(os.path.join(log_dir, 'app.log'), encoding='utf-8'), logging.StreamHandler(sys.stdout)])

if __name__ == '__main__':
    PROJECT_ROOT = find_project_root()
    setup_logging(PROJECT_ROOT)

    src_path = os.path.join(PROJECT_ROOT, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    
    settings = QSettings("GymPerformance", "AnalyzerApp")
    is_dark_mode = settings.value("dark_mode", False, type=bool)
    load_stylesheet(app, PROJECT_ROOT, dark=is_dark_mode)

    window = MainWindow(project_root=PROJECT_ROOT)
    window.show()
    sys.exit(app.exec_())