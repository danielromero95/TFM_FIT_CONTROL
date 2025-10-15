# src/gui/widgets/results_panel.py (Rediseñado)

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel, QGroupBox
from PyQt5.QtCore import Qt
from src import config
from .plot_widget import PlotWidget
from .video_player import VideoPlayerWidget # <-- NUEVO

class ResultsPanel(QWidget):
    """El panel que muestra todos los resultados del análisis en dos columnas."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Layout principal de dos columnas
        main_layout = QHBoxLayout(self)
        
        # --- COLUMNA IZQUIERDA: Reproductor de Vídeo ---
        self.video_player = VideoPlayerWidget()
        main_layout.addWidget(self.video_player, 1) # El 1 le da más espacio (stretch factor)

        # --- COLUMNA DERECHA: Gráfica y Métricas ---
        right_column_layout = QVBoxLayout()
        
        # Gráfica
        self.plot_widget = PlotWidget()
        right_column_layout.addWidget(self.plot_widget)

        # Métricas
        metrics_groupbox = QGroupBox("Resumen Biomecánico")
        self.metrics_layout = QFormLayout()
        metrics_groupbox.setLayout(self.metrics_layout)
        right_column_layout.addWidget(metrics_groupbox)
        
        main_layout.addLayout(right_column_layout, 1)

        # Labels para las métricas
        self.reps_label = QLabel("N/A")
        self.depth_label = QLabel("N/A")
        self.rom_label = QLabel("N/A")
        self.symmetry_label = QLabel("N/A")
        self.feedback_label = QLabel("Completa un análisis para ver los consejos.")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setAlignment(Qt.AlignTop)

        self.metrics_layout.addRow("Repeticiones Contadas:", self.reps_label)
        self.metrics_layout.addRow("Profundidad Media (Valle):", self.depth_label)
        self.metrics_layout.addRow("Rango de Movimiento (ROM):", self.rom_label)
        self.metrics_layout.addRow("Simetría de Rodillas:", self.symmetry_label)
        self.metrics_layout.addRow("Consejo del Experto:", self.feedback_label)

    def update_results(self, results):
        """Recibe el diccionario de resultados y actualiza toda la UI."""
        df = results.get('dataframe_metricas')
        rep_count = results.get('repeticiones_contadas')
        video_path = results.get('debug_video_path')

        # Cargar el vídeo si se ha generado
        if video_path:
            self.video_player.load_video(video_path)

        if df is None or df.empty: return

        # Actualizar gráfica y métricas
        self.plot_widget.plot_angle_series(df, 'rodilla_izq', config)
        
        angle_series = df['rodilla_izq'].dropna()
        symmetry_series = df['sim_rodilla'].dropna()

        profundidad = angle_series.min() if not angle_series.empty else 0
        rom = (angle_series.max() - profundidad) if not angle_series.empty else 0
        simetria = symmetry_series.mean() * 100 if not symmetry_series.empty else 100

        self.reps_label.setText(f"<b>{rep_count}</b>")
        self.depth_label.setText(f"<b>{profundidad:.1f}°</b>")
        self.rom_label.setText(f"<b>{rom:.1f}°</b>")
        self.symmetry_label.setText(f"<b>{simetria:.1f}%</b>")
        
        feedback = self.generate_feedback(profundidad, simetria)
        self.feedback_label.setText(f"<i>{feedback}</i>")

    def generate_feedback(self, depth, symmetry):
        # ... (lógica de feedback sin cambios) ...
        tips = []
        if depth > 110: tips.append("Intenta bajar un poco más para una sentadilla más profunda.")
        elif depth < 80: tips.append("¡Excelente profundidad!")
        if symmetry < 95: tips.append("Se detecta una asimetría entre tus rodillas.")
        return " ".join(tips) if tips else "¡Buen trabajo! La forma parece consistente."