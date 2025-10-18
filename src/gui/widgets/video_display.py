"""Drag-and-drop video thumbnail widget for the Qt app."""

from __future__ import annotations

import os
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class VideoDisplayWidget(QWidget):
    file_dropped = pyqtSignal(str)
    rotation_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.default_text = "Drag or select your video here"

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel(self.default_text, self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #777; font-size: 16px; background: transparent;")
        main_layout.addWidget(self.image_label, 1)

        self.controls_layout = QHBoxLayout()
        self.rotate_left_btn = QPushButton("↺ Rotate left")
        self.rotate_right_btn = QPushButton("Rotate right ↻")
        self.controls_layout.addStretch()
        self.controls_layout.addWidget(self.rotate_left_btn)
        self.controls_layout.addWidget(self.rotate_right_btn)
        self.controls_layout.addStretch()
        main_layout.addLayout(self.controls_layout)

        self.rotate_left_btn.clicked.connect(lambda: self.rotation_requested.emit(-90))
        self.rotate_right_btn.clicked.connect(lambda: self.rotation_requested.emit(90))

        self.normal_style = "VideoDisplayWidget { border: 2px dashed #aaa; border-radius: 8px; }"
        self.dragover_style = (
            "VideoDisplayWidget { border: 2px dashed #0078d7; border-radius: 8px; background-color: #e8f0fe; }"
        )
        self.setStyleSheet(self.normal_style)

        self.show_controls(False)

    def show_controls(self, show: bool) -> None:
        self.rotate_left_btn.setVisible(show)
        self.rotate_right_btn.setVisible(show)

    def dragEnterEvent(self, event):  # pragma: no cover - Qt callback
        if event.mimeData().hasUrls():
            self.setStyleSheet(self.dragover_style)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):  # pragma: no cover - Qt callback
        self.setStyleSheet(self.normal_style)

    def dropEvent(self, event):  # pragma: no cover - Qt callback
        self.setStyleSheet(self.normal_style)
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if os.path.isfile(path):
                self.file_dropped.emit(path)

    def set_thumbnail(self, pixmap):
        """Display ``pixmap`` and reveal the rotation controls."""
        self.image_label.setPixmap(pixmap)
        self.show_controls(True)

    def clear_content(self) -> None:
        """Reset the widget to its initial state."""
        self.image_label.clear()
        self.image_label.setText(self.default_text)
        self.show_controls(False)
