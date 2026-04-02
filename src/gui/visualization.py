from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QGridLayout
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from algorithms import ProgressData


class ScoreChart(pg.PlotWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setBackground('w')
        self.setTitle("Score Evolution", color='k')
        self.setLabel('left', 'Score', color='k')
        self.setLabel('bottom', 'Iteration', color='k')
        self.showGrid(x=True, y=True, alpha=0.3)

        self._current_curve = self.plot(pen=pg.mkPen('b', width=1), name='Current')
        self._best_curve = self.plot(pen=pg.mkPen('g', width=2), name='Best')

        self._iterations: list[int] = []
        self._current_scores: list[float] = []
        self._best_scores: list[float] = []

        legend = self.addLegend()
        legend.setOffset((10, 10))

    def add_point(self, iteration: int, current_score: float, best_score: float) -> None:
        self._iterations.append(iteration)
        self._current_scores.append(current_score)
        self._best_scores.append(best_score)

        self._current_curve.setData(self._iterations, self._current_scores)
        self._best_curve.setData(self._iterations, self._best_scores)

    def clear_data(self) -> None:
        self._iterations.clear()
        self._current_scores.clear()
        self._best_scores.clear()
        self._current_curve.setData([], [])
        self._best_curve.setData([], [])


class TemperatureChart(pg.PlotWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setBackground('w')
        self.setTitle("Temperature", color='k')
        self.setLabel('left', 'Temperature', color='k')
        self.setLabel('bottom', 'Iteration', color='k')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLogMode(y=True)

        self._temp_curve = self.plot(pen=pg.mkPen('r', width=2), name='Temperature')

        self._iterations: list[int] = []
        self._temperatures: list[float] = []

    def add_point(self, iteration: int, temperature: float) -> None:
        self._iterations.append(iteration)
        self._temperatures.append(max(temperature, 1e-10))
        self._temp_curve.setData(self._iterations, self._temperatures)

    def clear_data(self) -> None:
        self._iterations.clear()
        self._temperatures.clear()
        self._temp_curve.setData([], [])


class StatsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self._labels: dict[str, QLabel] = {}

        stats = [
            ("iteration", "Iteration:"),
            ("current_score", "Current Score:"),
            ("best_score", "Best Score:"),
            ("temperature", "Temperature:"),
            ("improvement", "Improvement:"),
        ]

        for row, (key, label_text) in enumerate(stats):
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold;")
            value = QLabel("-")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._labels[key] = value
            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)

        layout.setColumnStretch(1, 1)

    def update_stats(self, data: ProgressData, initial_score: float | None = None) -> None:
        self._labels["iteration"].setText(f"{data.iteration:,}")
        self._labels["current_score"].setText(f"{data.current_score:,.2f}")
        self._labels["best_score"].setText(f"{data.best_score:,.2f}")

        if "temperature" in data.extra:
            self._labels["temperature"].setText(f"{data.extra['temperature']:.4f}")
        else:
            self._labels["temperature"].setText("-")

        if initial_score is not None:
            improvement = initial_score - data.best_score
            pct = (improvement / initial_score) * 100 if initial_score > 0 else 0
            self._labels["improvement"].setText(f"{improvement:,.2f} ({pct:.2f}%)")
        else:
            self._labels["improvement"].setText("-")

    def clear_stats(self) -> None:
        for label in self._labels.values():
            label.setText("-")


class VisualizationPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._initial_score: float | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self._stats_panel = StatsPanel()
        stats_layout.addWidget(self._stats_panel)
        layout.addWidget(stats_group)

        self._score_chart = ScoreChart()
        self._score_chart.setMinimumHeight(200)
        layout.addWidget(self._score_chart, stretch=2)

        self._temp_chart = TemperatureChart()
        self._temp_chart.setMinimumHeight(150)
        layout.addWidget(self._temp_chart, stretch=1)

    def set_initial_score(self, score: float) -> None:
        self._initial_score = score

    def update_progress(self, data: ProgressData) -> None:
        self._score_chart.add_point(data.iteration, data.current_score, data.best_score)

        if "temperature" in data.extra:
            self._temp_chart.add_point(data.iteration, data.extra["temperature"])

        self._stats_panel.update_stats(data, self._initial_score)

    def clear(self) -> None:
        self._score_chart.clear_data()
        self._temp_chart.clear_data()
        self._stats_panel.clear_stats()
        self._initial_score = None
