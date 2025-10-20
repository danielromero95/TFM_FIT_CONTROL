"""Qt desktop application entry point."""

from __future__ import annotations

import os
import cv2
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QFont, QImage, QPixmap, QTransform
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from src import config
from src.A_preprocessing.video_metadata import get_video_rotation
from src.gui.style_utils import load_stylesheet
from src.gui.widgets.video_display import VideoDisplayWidget
from src.gui.worker import AnalysisWorker
from src.theme import load_theme_colors
from .widgets.results_panel import ResultsPanel

import logging

logger = logging.getLogger(__name__)


def _blend_hex(color: str, target: tuple[int, int, int], ratio: float) -> str:
    color_value = color.lstrip("#")
    if len(color_value) != 6:
        return color
    try:
        red = int(color_value[0:2], 16)
        green = int(color_value[2:4], 16)
        blue = int(color_value[4:6], 16)
    except ValueError:
        return color
    ratio = max(0.0, min(1.0, float(ratio)))
    blended = (
        int(round(red + (target[0] - red) * ratio)),
        int(round(green + (target[1] - green) * ratio)),
        int(round(blue + (target[2] - blue) * ratio)),
    )
    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


class MainWindow(QMainWindow):
    def __init__(self, project_root: str, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        self.video_path: str | None = None
        self.settings = QSettings(config.ORGANIZATION_NAME, config.APP_NAME)
        self.theme_colors = load_theme_colors(self.project_root)

        self.current_rotation = 0
        self.original_pixmap: QPixmap | None = None

        self.setWindowTitle(config.APP_NAME)
        self.resize(700, 650)
        self._init_ui()
        self._load_settings()

    def _init_ui(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_home_tab(), "Home")

        self.results_panel = ResultsPanel()
        self.tabs.addTab(self.results_panel, "Results")
        self.tabs.setTabEnabled(1, False)

        self.tabs.addTab(self._create_settings_tab(), "Settings")
        self.setCentralWidget(self.tabs)

    def _create_home_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        video_container = QWidget()
        video_container.setFixedHeight(400)
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)

        self.video_display = VideoDisplayWidget()
        self.video_display.file_dropped.connect(self._on_video_selected)
        self.video_display.rotation_requested.connect(self._on_rotation_requested)
        video_container_layout.addWidget(self.video_display)

        layout.addWidget(video_container)

        self.select_video_btn = QPushButton("Select video")
        self.select_video_btn.clicked.connect(self._open_file_dialog)
        layout.addWidget(self.select_video_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.results_label = QLabel("Analysis results will appear here.")
        self.results_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.results_label.setFont(font)
        self.results_label.setStyleSheet(
            "color: #333; padding: 10px; border-radius: 5px; background-color: #f0f0f0;"
        )
        layout.addWidget(self.results_label)

        self.process_btn = QPushButton("Analyze video")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._start_analysis)
        self._apply_primary_button_style(self.process_btn)
        layout.addWidget(self.process_btn)

        return widget

    def _apply_primary_button_style(self, button: QPushButton) -> None:
        primary = self.theme_colors.primary
        hover = _blend_hex(primary, (255, 255, 255), 0.12)
        pressed = _blend_hex(primary, (0, 0, 0), 0.1)
        disabled = _blend_hex(primary, (255, 255, 255), 0.45)
        button.setStyleSheet(
            (
                "QPushButton {"
                f"background-color: {primary};"
                "color: #ffffff;"
                "border: none;"
                "padding: 10px 18px;"
                "border-radius: 10px;"
                "font-weight: 600;"
                "}"
                "QPushButton:hover {"
                f"background-color: {hover};"
                "}"
                "QPushButton:pressed {"
                f"background-color: {pressed};"
                "}"
                "QPushButton:disabled {"
                f"background-color: {disabled};"
                "color: rgba(255, 255, 255, 0.7);"
                "}"
            )
        )

    def _create_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)

        self.output_dir_edit = QLineEdit()
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setMinimum(1)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(16, 4096)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(16, 4096)

        self.use_crop_check = QCheckBox("Use centred crop (more precise)")
        self.generate_video_check = QCheckBox("Generate debug video with skeleton")
        self.debug_mode_check = QCheckBox("Debug mode (saves intermediate CSVs)")
        self.dark_mode_check = QCheckBox("Dark mode")
        self.dark_mode_check.stateChanged.connect(self._toggle_theme)

        size_layout = QHBoxLayout()
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.height_spin)

        layout.addRow("Base output folder:", self.output_dir_edit)
        layout.addRow("Sample rate (1 of every N frames):", self.sample_rate_spin)
        layout.addRow("Preprocessing width/height (px):", size_layout)
        layout.addRow(self.use_crop_check)
        layout.addRow(self.generate_video_check)
        layout.addRow(self.debug_mode_check)
        layout.addRow(self.dark_mode_check)

        return widget

    def _toggle_theme(self, state: int) -> None:
        is_dark = state == Qt.Checked
        load_stylesheet(QApplication.instance(), self.project_root, is_dark)

    def _set_processing_state(self, is_processing: bool) -> None:
        is_enabled = not is_processing
        self.tabs.setTabEnabled(1, is_enabled)
        self.select_video_btn.setEnabled(is_enabled)
        self.video_display.show_controls(is_enabled)
        self.process_btn.setEnabled(is_enabled and self.video_path is not None)
        if is_processing:
            self.results_label.setText("Processing... please wait.")
            self.results_label.setStyleSheet(
                "color: #f39c12; padding: 10px; border-radius: 5px; background-color: #fef9e7;"
            )

    def _on_video_selected(self, path: str) -> None:
        self.video_path = path
        try:
            auto_rotation = get_video_rotation(path)
            self.current_rotation = auto_rotation
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Rotation autodetection failed: %s", exc)
            self.current_rotation = 0

        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.original_pixmap = QPixmap.fromImage(
                QImage(
                    frame_rgb.data,
                    frame_rgb.shape[1],
                    frame_rgb.shape[0],
                    frame_rgb.strides[0],
                    QImage.Format_RGB888,
                )
            )
            transform = QTransform().rotate(self.current_rotation)
            rotated_pixmap = self.original_pixmap.transformed(transform)
            self.video_display.set_thumbnail(
                rotated_pixmap.scaled(
                    self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

        self.process_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.results_label.setText("Video loaded. Ready to analyse.")
        self.results_label.setStyleSheet(
            "color: #333; padding: 10px; border-radius: 5px; background-color: #f0f0f0;"
        )

    def _on_rotation_requested(self, angle: int) -> None:
        if self.original_pixmap is None:
            return
        self.current_rotation = (self.current_rotation + angle) % 360
        logger.info("Manual thumbnail rotation set to %d degrees.", self.current_rotation)

        transform = QTransform().rotate(self.current_rotation)
        rotated_pixmap = self.original_pixmap.transformed(transform)
        self.video_display.set_thumbnail(
            rotated_pixmap.scaled(
                self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _start_analysis(self) -> None:
        if not self.video_path:
            QMessageBox.critical(self, "Error", "No video has been selected.")
            return

        cfg = config.load_default()
        base_dir = Path(self.output_dir_edit.text().strip()).expanduser()
        cfg.output.base_dir = base_dir
        cfg.output.counts_dir = base_dir / "counts"
        cfg.output.poses_dir = base_dir / "poses"
        cfg.video.manual_sample_rate = int(self.sample_rate_spin.value())
        cfg.pose.rotate = self.current_rotation
        cfg.pose.target_width = int(self.width_spin.value())
        cfg.pose.target_height = int(self.height_spin.value())
        cfg.pose.use_crop = self.use_crop_check.isChecked()
        cfg.debug.generate_debug_video = self.generate_video_check.isChecked()
        cfg.debug.debug_mode = self.debug_mode_check.isChecked()

        self.worker = AnalysisWorker(self.video_path, cfg)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self._on_processing_error)
        self.worker.finished.connect(self._on_processing_finished)
        self.worker.finished.connect(lambda: self._set_processing_state(False))

        self._set_processing_state(True)
        self.worker.start()

    def _on_processing_error(self, error_message: str) -> None:
        self.results_label.setText(f"Error: {error_message}")
        self.results_label.setStyleSheet(
            "color: #c0392b; padding: 10px; border-radius: 5px; background-color: #fdedec;"
        )
        QMessageBox.critical(self, "Processing error", error_message)

    def _on_processing_finished(self, results: dict) -> None:
        rep_count = results.get("repetition_count", "N/A")
        self.results_label.setText(f"Analysis complete! Repetitions detected: {rep_count}")
        self.results_label.setStyleSheet(
            "color: #27ae60; padding: 10px; border-radius: 5px; background-color: #eafaf1;"
        )
        QMessageBox.information(
            self,
            "Finished",
            f"Analysis completed.\n\nRepetitions counted: {rep_count}",
        )

        self.results_panel.update_results(results)
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentWidget(self.results_panel)

    def _load_settings(self) -> None:
        self.output_dir_edit.setText(
            self.settings.value("output_dir", os.path.join(self.project_root, "data", "processed"))
        )
        self.sample_rate_spin.setValue(
            self.settings.value("sample_rate", config.DEFAULT_SAMPLE_RATE, type=int)
        )
        self.width_spin.setValue(
            self.settings.value("width", config.DEFAULT_TARGET_WIDTH, type=int)
        )
        self.height_spin.setValue(
            self.settings.value("height", config.DEFAULT_TARGET_HEIGHT, type=int)
        )
        self.use_crop_check.setChecked(
            self.settings.value("use_crop", config.DEFAULT_USE_CROP, type=bool)
        )
        self.generate_video_check.setChecked(
            self.settings.value("generate_debug_video", config.DEFAULT_GENERATE_VIDEO, type=bool)
        )
        self.debug_mode_check.setChecked(
            self.settings.value("debug_mode", config.DEFAULT_DEBUG_MODE, type=bool)
        )
        is_dark = self.settings.value("dark_mode", config.DEFAULT_DARK_MODE, type=bool)
        self.dark_mode_check.setChecked(is_dark)
        self._toggle_theme(Qt.Checked if is_dark else Qt.Unchecked)

    def closeEvent(self, event):  # pragma: no cover - Qt callback
        self.settings.setValue("output_dir", self.output_dir_edit.text())
        self.settings.setValue("sample_rate", self.sample_rate_spin.value())
        self.settings.setValue("width", self.width_spin.value())
        self.settings.setValue("height", self.height_spin.value())
        self.settings.setValue("use_crop", self.use_crop_check.isChecked())
        self.settings.setValue("generate_debug_video", self.generate_video_check.isChecked())
        self.settings.setValue("debug_mode", self.debug_mode_check.isChecked())
        self.settings.setValue("dark_mode", self.dark_mode_check.isChecked())
        super().closeEvent(event)

    def _open_file_dialog(self) -> None:
        default_input = (
            os.path.dirname(self.video_path)
            if self.video_path
            else os.path.join(self.project_root, "data", "raw")
        )

        wildcard_extensions = [f"*{ext}" for ext in config.VIDEO_EXTENSIONS]
        filter_string = f"Videos ({' '.join(wildcard_extensions)})"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select video", default_input, filter_string
        )
        if file_path:
            self._on_video_selected(file_path)
