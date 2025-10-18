"""Matplotlib plot widget used inside the Qt desktop app."""

from __future__ import annotations

from PyQt5.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotWidget(QWidget):
    """Embed a Matplotlib figure within a PyQt widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_angle_series(self, df_metrics, angle_column: str, config_module):
        """Plot the raw and smoothed angle series along with thresholds."""
        self.ax.clear()

        raw_series = df_metrics[angle_column]
        smooth_series = raw_series.rolling(window=5, center=True, min_periods=1).mean()
        frames = df_metrics["frame_idx"]

        self.ax.plot(frames, raw_series, label="Angle (raw)", alpha=0.4)
        self.ax.plot(frames, smooth_series, label="Angle (smoothed)", color="red", linewidth=2)

        self.ax.axhline(
            y=config_module.SQUAT_HIGH_THRESH,
            color="g",
            linestyle="--",
            label=f"High threshold ({config_module.SQUAT_HIGH_THRESH}°)",
        )
        self.ax.axhline(
            y=config_module.SQUAT_LOW_THRESH,
            color="orange",
            linestyle="--",
            label=f"Low threshold ({config_module.SQUAT_LOW_THRESH}°)",
        )

        self.ax.set_title("Knee angle analysis")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Angle (degrees)")
        self.ax.legend()
        self.ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()
