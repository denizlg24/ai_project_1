from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter, QMessageBox, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from core import Problem
from algorithms import Algorithm, ProgressData
from .config_panel import ConfigPanel
from .visualization import VisualizationPanel


class AlgorithmWorker(QObject):
    progress = pyqtSignal(ProgressData)
    finished = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, algorithm: Algorithm):
        super().__init__()
        self._algorithm = algorithm

    def run(self) -> None:
        try:
            result = self._algorithm.run(progress_callback=self._on_progress)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, data: ProgressData) -> None:
        self.progress.emit(data)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._problem: Problem | None = None
        self._algorithm: Algorithm | None = None
        self._worker: AlgorithmWorker | None = None
        self._thread: QThread | None = None
        self._data_file: str | None = None
        self._submission_file: str | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Optimization Algorithms")
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._config_panel = ConfigPanel()
        self._config_panel.setMinimumWidth(300)
        self._config_panel.setMaximumWidth(400)
        splitter.addWidget(self._config_panel)

        self._viz_panel = VisualizationPanel()
        splitter.addWidget(self._viz_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _connect_signals(self) -> None:
        self._config_panel.data_file_changed.connect(self._on_data_file_changed)
        self._config_panel.submission_file_changed.connect(self._on_submission_file_changed)
        self._config_panel.run_requested.connect(self._on_run)
        self._config_panel.stop_requested.connect(self._on_stop)

    def _on_data_file_changed(self, path: str) -> None:
        self._data_file = path
        self._status_bar.showMessage(f"Data file: {path}")

    def _on_submission_file_changed(self, path: str) -> None:
        self._submission_file = path
        self._status_bar.showMessage(f"Submission file: {path}")

    def _on_run(self) -> None:
        if not self._data_file:
            QMessageBox.warning(self, "No Data File", "Please select a data file first.")
            return

        algo_cls = self._config_panel.get_algorithm_class()
        if algo_cls is None:
            QMessageBox.warning(self, "No Algorithm", "Please select an algorithm.")
            return

        try:
            self._problem = Problem.from_files(self._data_file, self._submission_file)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Problem", str(e))
            return

        initial_score = self._problem.total_score()
        self._viz_panel.clear()
        self._viz_panel.set_initial_score(initial_score)

        self._algorithm = algo_cls(self._problem)
        params = self._config_panel.get_parameters()
        self._algorithm.configure(**params)

        self._config_panel.set_running(True)
        self._status_bar.showMessage(f"Running {algo_cls.name()}...")

        self._thread = QThread()
        self._worker = AlgorithmWorker(self._algorithm)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _on_stop(self) -> None:
        if self._algorithm:
            self._algorithm.request_stop()
            self._status_bar.showMessage("Stopping...")

    def _on_progress(self, data: ProgressData) -> None:
        self._viz_panel.update_progress(data)

    def _on_finished(self, score: float) -> None:
        self._config_panel.set_running(False)
        self._status_bar.showMessage(f"Finished. Best score: {score:,.2f}")

        self._cleanup_thread()

    def _on_error(self, message: str) -> None:
        self._config_panel.set_running(False)
        self._status_bar.showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Algorithm Error", message)

        self._cleanup_thread()

    def _cleanup_thread(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

    def closeEvent(self, event) -> None:
        if self._algorithm:
            self._algorithm.request_stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        event.accept()
