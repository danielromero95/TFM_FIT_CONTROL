# src/gui/widgets/plot_widget.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotWidget(QWidget):
    """Un widget para incrustar una gráfica de Matplotlib en PyQt."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_angle_series(self, df_metrics, angle_column, config):
        """Dibuja la serie de ángulos con sus umbrales."""
        self.ax.clear()
        
        # Extraer datos
        raw_series = df_metrics[angle_column]
        smooth_series = raw_series.rolling(window=5, center=True, min_periods=1).mean()
        frames = df_metrics['frame_idx']
        
        # Dibujar series
        self.ax.plot(frames, raw_series, label='Ángulo (Raw)', alpha=0.4)
        self.ax.plot(frames, smooth_series, label='Ángulo (Suavizado)', color='red', linewidth=2)
        
        # Dibujar umbrales
        self.ax.axhline(y=config.SQUAT_HIGH_THRESH, color='g', linestyle='--', label=f'Umbral Alto ({config.SQUAT_HIGH_THRESH}°)')
        self.ax.axhline(y=config.SQUAT_LOW_THRESH, color='orange', linestyle='--', label=f'Umbral Bajo ({config.SQUAT_LOW_THRESH}°)')
        
        # Estilo
        self.ax.set_title('Análisis de Ángulo de Rodilla')
        self.ax.set_xlabel('Fotograma')
        self.ax.set_ylabel('Ángulo (grados)')
        self.ax.legend()
        self.ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()