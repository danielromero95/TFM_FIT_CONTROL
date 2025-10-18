"""Helpers to load Qt stylesheets."""

import os
from PyQt5.QtWidgets import QApplication


def load_stylesheet(app: QApplication, project_root: str, dark: bool) -> None:
    """Load the light or dark stylesheet for the application."""
    themes_dir = os.path.join(project_root, "themes")
    qss_file = "dark.qss" if dark else "light.qss"
    path = os.path.join(themes_dir, qss_file)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            app.setStyleSheet(handle.read())
    else:
        print(f"Warning: stylesheet not found at {path}")
