# src/gui/worker.py
import logging
from PyQt5.QtCore import QThread, pyqtSignal
from src.pipeline import run_full_pipeline_in_memory

logger = logging.getLogger(__name__)

class AnalysisWorker(QThread):
    """Ejecuta el pipeline de análisis en un hilo separado."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, video_path, settings, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.settings = settings

    def run(self):
        try:
            results = run_full_pipeline_in_memory(
                video_path=self.video_path,
                settings=self.settings,
                progress_callback=self.progress.emit
            )
            self.finished.emit(results)
        except Exception as e:
            logger.exception("Error durante la ejecución del pipeline en el WorkerThread")
            self.error.emit(str(e))