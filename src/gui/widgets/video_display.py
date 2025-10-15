# src/gui/widgets/video_display.py
import os
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal

class VideoDisplayWidget(QWidget):
    file_dropped = pyqtSignal(str)
    rotation_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.default_text = "Arrastra o selecciona tu vídeo aquí"
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)

        self.image_label = QLabel(self.default_text, self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #777; font-size: 16px; background: transparent;")
        main_layout.addWidget(self.image_label, 1) # El '1' le da stretch

        self.controls_layout = QHBoxLayout()
        self.rotate_left_btn = QPushButton("↺ Girar Izquierda")
        self.rotate_right_btn = QPushButton("Girar Derecha ↻")
        self.controls_layout.addStretch()
        self.controls_layout.addWidget(self.rotate_left_btn)
        self.controls_layout.addWidget(self.rotate_right_btn)
        self.controls_layout.addStretch()
        main_layout.addLayout(self.controls_layout)
        
        self.rotate_left_btn.clicked.connect(lambda: self.rotation_requested.emit(-90))
        self.rotate_right_btn.clicked.connect(lambda: self.rotation_requested.emit(90))

        self.normal_style = "VideoDisplayWidget { border: 2px dashed #aaa; border-radius: 8px; }"
        self.dragover_style = "VideoDisplayWidget { border: 2px dashed #0078d7; border-radius: 8px; background-color: #e8f0fe; }"
        self.setStyleSheet(self.normal_style)
        
        self.show_controls(False)

    def show_controls(self, show: bool):
        self.rotate_left_btn.setVisible(show)
        self.rotate_right_btn.setVisible(show)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.setStyleSheet(self.dragover_style); event.acceptProposedAction()
    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.normal_style)
    def dropEvent(self, event):
        self.setStyleSheet(self.normal_style)
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if os.path.isfile(path): self.file_dropped.emit(path)

    def set_thumbnail(self, pixmap):
        # --- CAMBIO CLAVE: Simplificado para no interferir con el layout ---
        self.image_label.setPixmap(pixmap)
        self.show_controls(True)

    def clear_content(self):
        self.image_label.clear()
        self.image_label.setText(self.default_text)
        self.show_controls(False)