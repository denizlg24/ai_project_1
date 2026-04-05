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


METRIC_CONFIG: dict[str, dict[str, str | bool]] = {
    "temperature": {
        "title": "Temperature",
        "label": "Temperature",
        "color": "r",
        "log_scale": True,
    },
    "population_diversity": {
        "title": "Population Diversity",
        "label": "Std Dev of Scores",
        "color": "m",
        "log_scale": False,
    },
    "neighbourhood": {
        "title": "Active Neighbourhood",
        "label": "Neighbourhood (k)",
        "color": "c",
        "log_scale": False,
    },
}


class ExtraMetricChart(pg.PlotWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setBackground('w')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('bottom', 'Iteration', color='k')

        self._curve = self.plot(pen=pg.mkPen('r', width=2))
        self._iterations: list[int] = []
        self._values: list[float] = []
        self._current_metric: str | None = None
        self.hide()

    def configure_metric(self, metric_name: str) -> None:
        if metric_name == self._current_metric:
            return
        self._current_metric = metric_name
        config = METRIC_CONFIG.get(metric_name, {})
        title = str(config.get("title", metric_name))
        label = str(config.get("label", metric_name))
        color = str(config.get("color", "r"))
        log_scale = bool(config.get("log_scale", False))

        self.setTitle(title, color='k')
        self.setLabel('left', label, color='k')
        self.setLogMode(y=log_scale)
        self._curve.setPen(pg.mkPen(color, width=2))
        self.show()

    def add_point(self, iteration: int, value: float) -> None:
        self._iterations.append(iteration)
        self._values.append(max(value, 1e-10))
        self._curve.setData(self._iterations, self._values)

    def clear_data(self) -> None:
        self._iterations.clear()
        self._values.clear()
        self._curve.setData([], [])
        self._current_metric = None
        self.hide()


class StatsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self._labels: dict[str, QLabel] = {}
        self._label_names: dict[str, QLabel] = {}

        stats = [
            ("iteration", "Iteration:"),
            ("elapsed", "Elapsed:"),
            ("current_score", "Current Score:"),
            ("best_score", "Best Score:"),
            ("extra_metric", "Extra:"),
            ("improvement", "Improvement:"),
        ]

        for row, (key, label_text) in enumerate(stats):
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold;")
            value = QLabel("-")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._labels[key] = value
            self._label_names[key] = label
            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)

        layout.setColumnStretch(1, 1)

    def update_stats(self, data: ProgressData, initial_score: float | None = None) -> None:
        self._labels["iteration"].setText(f"{data.iteration:,}")

        mins, secs = divmod(int(data.elapsed_seconds), 60)
        self._labels["elapsed"].setText(f"{mins}m {secs:02d}s")

        self._labels["current_score"].setText(f"{data.current_score:,.2f}")
        self._labels["best_score"].setText(f"{data.best_score:,.2f}")

        if data.extra:
            metric_name = next(iter(data.extra))
            config = METRIC_CONFIG.get(metric_name, {})
            display_name = str(config.get("title", metric_name))
            self._label_names["extra_metric"].setText(f"{display_name}:")
            self._labels["extra_metric"].setText(f"{data.extra[metric_name]:,.4f}")
        else:
            self._label_names["extra_metric"].setText("Extra:")
            self._labels["extra_metric"].setText("-")

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

        self._extra_chart = ExtraMetricChart()
        self._extra_chart.setMinimumHeight(150)
        layout.addWidget(self._extra_chart, stretch=1)

    def set_initial_score(self, score: float) -> None:
        self._initial_score = score

    def update_progress(self, data: ProgressData) -> None:
        self._score_chart.add_point(data.iteration, data.current_score, data.best_score)

        if data.extra:
            metric_name = next(iter(data.extra))
            self._extra_chart.configure_metric(metric_name)
            self._extra_chart.add_point(data.iteration, data.extra[metric_name])

        self._stats_panel.update_stats(data, self._initial_score)

    def clear(self) -> None:
        self._score_chart.clear_data()
        self._extra_chart.clear_data()
        self._stats_panel.clear_stats()
        self._initial_score = None
