"""Worker thread used by the Qt GUI to execute the pipeline."""

from __future__ import annotations

import logging
from PyQt5.QtCore import QThread, pyqtSignal

from src import config
from src.pipeline import Report, run_pipeline

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """Execute the analysis pipeline on a background thread."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, video_path: str, cfg: config.Config, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cfg = cfg.copy()

    def run(self):
        try:
            report: Report = run_pipeline(
                video_path=self.video_path,
                cfg=self.cfg,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(report.to_legacy_dict())
        except Exception as exc:  # pragma: no cover - surfaced to the GUI user
            logger.exception("Error while running the pipeline inside WorkerThread")
            self.error.emit(str(exc))
