# src/gui/main_window.py

import os
import cv2
import logging
from PyQt5.QtWidgets import (QMainWindow, QWidget, QTabWidget, QVBoxLayout, QFormLayout, 
                             QHBoxLayout, QPushButton, QProgressBar, QLabel, QSpinBox, 
                             QComboBox, QCheckBox, QLineEdit, QFileDialog, QMessageBox, QApplication)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QPixmap, QImage, QFont, QTransform

from src import config
from src.gui.style_utils import load_stylesheet
from src.gui.widgets.video_display import VideoDisplayWidget
from .widgets.results_panel import ResultsPanel
from src.gui.worker import AnalysisWorker

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self, project_root, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        self.video_path = None
        self.settings = QSettings(config.ORGANIZATION_NAME, config.APP_NAME)
        
        self.current_rotation = 0
        self.original_pixmap = None

        self.setWindowTitle(config.APP_NAME)
        self.resize(700, 650)
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_home_tab(), "Inicio")

        # --- Creamos el panel de resultados y lo añadimos como pestaña ---
        self.results_panel = ResultsPanel()
        self.tabs.addTab(self.results_panel, "Resultados")
        self.tabs.setTabEnabled(1, False) # Deshabilitada hasta que haya resultados

        self.tabs.addTab(self._create_settings_tab(), "Ajustes")
        self.setCentralWidget(self.tabs)

    def _create_home_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # --- Contenedor para la preview con altura fija ---
        video_container = QWidget()
        video_container.setFixedHeight(400) # El contenedor tiene la altura fija
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0,0,0,0)
        
        # El widget de la imagen va dentro del contenedor
        self.video_display = VideoDisplayWidget()
        self.video_display.file_dropped.connect(self._on_video_selected)
        self.video_display.rotation_requested.connect(self._on_rotation_requested)
        video_container_layout.addWidget(self.video_display)
        
        # Añadimos el contenedor al layout principal de la pestaña
        layout.addWidget(video_container)

        # --- Resto de widgets ---
        self.select_video_btn = QPushButton("Seleccionar vídeo")
        self.select_video_btn.clicked.connect(self._open_file_dialog)
        layout.addWidget(self.select_video_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.results_label = QLabel("Los resultados del análisis aparecerán aquí.")
        self.results_label.setAlignment(Qt.AlignCenter)
        font = QFont(); font.setPointSize(14); font.setBold(True)
        self.results_label.setFont(font)
        self.results_label.setStyleSheet("color: #333; padding: 10px; border-radius: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.results_label)
        
        self.process_btn = QPushButton("Analizar Vídeo")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._start_analysis)
        layout.addWidget(self.process_btn)
        
        return widget

    def _create_settings_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.output_dir_edit = QLineEdit()
        self.sample_rate_spin = QSpinBox(); self.sample_rate_spin.setMinimum(1)
        self.width_spin = QSpinBox(); self.width_spin.setRange(16,4096)
        self.height_spin = QSpinBox(); self.height_spin.setRange(16,4096)
        
        self.use_crop_check = QCheckBox("Usar recorte centrado (más preciso)")
        self.generate_video_check = QCheckBox("Generar vídeo de depuración con esqueleto")
        self.debug_mode_check = QCheckBox("Modo Depuración (guarda CSVs intermedios)")
        self.dark_mode_check = QCheckBox("Modo oscuro")
        self.dark_mode_check.stateChanged.connect(self._toggle_theme)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.width_spin)
        h_layout.addWidget(QLabel("x"))
        h_layout.addWidget(self.height_spin)

        layout.addRow("Carpeta base de salida:", self.output_dir_edit)
        layout.addRow("Sample Rate (1 de cada N frames):", self.sample_rate_spin)
        layout.addRow("Ancho/Alto (px) de preproceso:", h_layout)
        layout.addRow(self.use_crop_check)
        layout.addRow(self.generate_video_check)
        layout.addRow(self.debug_mode_check)
        layout.addRow(self.dark_mode_check)
        
        return widget
    
    def _toggle_theme(self, state):
        is_dark = (state == Qt.Checked)
        load_stylesheet(QApplication.instance(), self.project_root, dark=is_dark)

    def _on_video_selected(self, path):
        self.video_path = path
        self.current_rotation = 0
        self.results_label.setText("Vídeo cargado. Listo para analizar.")
        self.results_label.setStyleSheet("color: #0057e7; padding: 10px; border-radius: 5px; background-color: #e8f0fe;")
        
        try:
            from src.A_preprocessing.video_metadata import get_video_rotation
            auto_rotation = get_video_rotation(path)
            self.current_rotation = auto_rotation
        except Exception as e:
            logger.error(f"Fallo en la autodetección de rotación: {e}")
            self.current_rotation = 0

        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.original_pixmap = QPixmap.fromImage(QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format_RGB888))
            
            transform = QTransform().rotate(self.current_rotation)
            rotated_pixmap = self.original_pixmap.transformed(transform)
            self.video_display.set_thumbnail(rotated_pixmap.scaled(self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.process_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
    def _on_rotation_requested(self, angle):
        if self.original_pixmap is None:
            return
            
        self.current_rotation = (self.current_rotation + angle) % 360
        logger.info(f"Rotación manual del thumbnail a {self.current_rotation} grados.")

        transform = QTransform().rotate(self.current_rotation)
        rotated_pixmap = self.original_pixmap.transformed(transform)
        
        self.video_display.set_thumbnail(rotated_pixmap.scaled(self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _start_analysis(self):
        if not self.video_path:
            QMessageBox.critical(self, "Error", "No se ha seleccionado ningún vídeo.")
            return

        settings = {
            'output_dir': self.output_dir_edit.text().strip(),
            'sample_rate': self.sample_rate_spin.value(),
            'rotate': self.current_rotation,
            'target_width': self.width_spin.value(),
            'target_height': self.height_spin.value(),
            'use_crop': self.use_crop_check.isChecked(),
            'generate_debug_video': self.generate_video_check.isChecked(),
            'debug_mode': self.debug_mode_check.isChecked()
        }
        
        self.worker = AnalysisWorker(self.video_path, settings)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self._on_processing_error)
        self.worker.finished.connect(self._on_processing_finished)
        self.worker.finished.connect(lambda: self._set_processing_state(False))
        
        self._set_processing_state(True)
        self.worker.start()
        
    def _set_processing_state(self, is_processing):
        is_enabled = not is_processing
        self.tabs.setTabEnabled(1, is_enabled)
        self.select_video_btn.setEnabled(is_enabled)
        self.video_display.show_controls(is_enabled)
        self.process_btn.setEnabled(is_enabled and self.video_path is not None)
        if is_processing:
            self.results_label.setText("Procesando... por favor, espere.")
            self.results_label.setStyleSheet("color: #f39c12; padding: 10px; border-radius: 5px; background-color: #fef9e7;")

    def _on_processing_error(self, error_message):
        self.results_label.setText(f"Error: {error_message}")
        self.results_label.setStyleSheet("color: #c0392b; padding: 10px; border-radius: 5px; background-color: #fdedec;")
        QMessageBox.critical(self, "Error de Procesamiento", error_message)

    def _on_processing_finished(self, results):
        rep_count = results.get("repeticiones_contadas", "N/A")
        self.results_label.setText(f"¡Análisis Completo! Repeticiones detectadas: {rep_count}")
        self.results_label.setStyleSheet("color: #27ae60; padding: 10px; border-radius: 5px; background-color: #eafaf1;")
        QMessageBox.information(self, "Finalizado", f"Análisis completado.\n\nRepeticiones contadas: {rep_count}")

        # ---  Actualizamos el panel de resultados y lo mostramos ---
        self.results_panel.update_results(results)
        self.tabs.setTabEnabled(1, True) # Habilitamos la pestaña
        self.tabs.setCurrentWidget(self.results_panel) # Cambiamos a la pestaña de resultados

    
    def _load_settings(self):
        self.output_dir_edit.setText(self.settings.value("output_dir", os.path.join(self.project_root, 'data', 'processed')))
        self.sample_rate_spin.setValue(self.settings.value("sample_rate", config.DEFAULT_SAMPLE_RATE, type=int))
        self.width_spin.setValue(self.settings.value("width", config.DEFAULT_TARGET_WIDTH, type=int))
        self.height_spin.setValue(self.settings.value("height", config.DEFAULT_TARGET_HEIGHT, type=int))
        self.use_crop_check.setChecked(self.settings.value("use_crop", config.DEFAULT_USE_CROP, type=bool))
        self.generate_video_check.setChecked(self.settings.value("generate_debug_video", config.DEFAULT_GENERATE_VIDEO, type=bool))
        self.debug_mode_check.setChecked(self.settings.value("debug_mode", config.DEFAULT_DEBUG_MODE, type=bool))
        is_dark = self.settings.value("dark_mode", config.DEFAULT_DARK_MODE, type=bool)
        self.dark_mode_check.setChecked(is_dark)
        self._toggle_theme(Qt.Checked if is_dark else Qt.Unchecked)

    def closeEvent(self, event):
        self.settings.setValue("output_dir", self.output_dir_edit.text())
        self.settings.setValue("sample_rate", self.sample_rate_spin.value())
        self.settings.setValue("width", self.width_spin.value())
        self.settings.setValue("height", self.height_spin.value())
        self.settings.setValue("use_crop", self.use_crop_check.isChecked())
        self.settings.setValue("generate_debug_video", self.generate_video_check.isChecked())
        self.settings.setValue("debug_mode", self.debug_mode_check.isChecked())
        self.settings.setValue("dark_mode", self.dark_mode_check.isChecked())
        super().closeEvent(event)

    def _open_file_dialog(self):
        default_input = os.path.dirname(self.video_path) if self.video_path else os.path.join(self.project_root, 'data', 'raw')
        
        # --- CORRECCIÓN: Construimos el filtro correctamente ---
        # Creamos una lista de patrones con comodín, ej: ["*.mp4", "*.mov"]
        wildcard_extensions = [f"*{ext}" for ext in config.VIDEO_EXTENSIONS]
        # Unimos los patrones con un espacio
        filter_string = f"Vídeos ({' '.join(wildcard_extensions)})"
        
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar vídeo", default_input, filter_string)
        if file:
            self._on_video_selected(file)