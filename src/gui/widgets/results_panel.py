"""Results panel shown in the Qt desktop application."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src import config
from .plot_widget import PlotWidget
from .video_player import VideoPlayerWidget


class ResultsPanel(QWidget):
    """Display analysis outputs side-by-side."""

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QHBoxLayout(self)

        # Left column: video player.
        self.video_player = VideoPlayerWidget()
        main_layout.addWidget(self.video_player, 1)

        # Right column: plot and metrics summary.
        right_column_layout = QVBoxLayout()

        self.plot_widget = PlotWidget()
        right_column_layout.addWidget(self.plot_widget)

        metrics_groupbox = QGroupBox("Biomechanics summary")
        self.metrics_layout = QFormLayout()
        metrics_groupbox.setLayout(self.metrics_layout)
        right_column_layout.addWidget(metrics_groupbox)

        main_layout.addLayout(right_column_layout, 1)

        # Metric labels.
        self.reps_label = QLabel("N/A")
        self.depth_label = QLabel("N/A")
        self.rom_label = QLabel("N/A")
        self.symmetry_label = QLabel("N/A")
        self.feedback_label = QLabel("Run an analysis to see recommendations.")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setAlignment(Qt.AlignTop)

        self.metrics_layout.addRow("Repetitions counted:", self.reps_label)
        self.metrics_layout.addRow("Average depth (valley):", self.depth_label)
        self.metrics_layout.addRow("Range of motion (ROM):", self.rom_label)
        self.metrics_layout.addRow("Knee symmetry:", self.symmetry_label)
        self.metrics_layout.addRow("Coach tip:", self.feedback_label)

    def update_results(self, results):
        """Update the panel with a ``results`` mapping."""
        df = results.get("metrics_dataframe")
        rep_count = results.get("repetition_count")
        video_path = results.get("debug_video_path")

        if video_path:
            self.video_player.load_video(video_path)

        if df is None or df.empty:
            return

        self.plot_widget.plot_angle_series(df, "left_knee", config)

        angle_series = df["left_knee"].dropna()
        symmetry_series = df["knee_symmetry"].dropna()

        depth = angle_series.min() if not angle_series.empty else 0
        rom = (angle_series.max() - depth) if not angle_series.empty else 0
        symmetry = symmetry_series.mean() * 100 if not symmetry_series.empty else 100

        self.reps_label.setText(f"<b>{rep_count}</b>")
        self.depth_label.setText(f"<b>{depth:.1f}°</b>")
        self.rom_label.setText(f"<b>{rom:.1f}°</b>")
        self.symmetry_label.setText(f"<b>{symmetry:.1f}%</b>")

        feedback = self.generate_feedback(depth, symmetry)
        self.feedback_label.setText(f"<i>{feedback}</i>")

    def generate_feedback(self, depth: float, symmetry: float) -> str:
        """Produce a short, human-readable recommendation."""
        tips: list[str] = []
        if depth > 110:
            tips.append("Try to squat a bit deeper for improved range.")
        elif depth < 80:
            tips.append("Great depth!")
        if symmetry < 95:
            tips.append("Asymmetry detected between knees.")
        return " ".join(tips) if tips else "Nice work! Your form looks consistent."
