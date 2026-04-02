from typing import Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QFormLayout, QPushButton, QFileDialog
)
from PyQt6.QtCore import pyqtSignal

from algorithms import Algorithm, ParameterDef, SimulatedAnnealing, GeneticAlgorithm


ALGORITHM_REGISTRY: list[type[Algorithm]] = [
    SimulatedAnnealing,
    GeneticAlgorithm
]


class ConfigPanel(QWidget):
    algorithm_changed = pyqtSignal(type)
    parameters_changed = pyqtSignal(dict)
    data_file_changed = pyqtSignal(str)
    submission_file_changed = pyqtSignal(str)
    run_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._param_widgets: dict[str, QWidget] = {}
        self._current_algorithm: type[Algorithm] | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        file_group = QGroupBox("Problem Instance")
        file_layout = QFormLayout(file_group)

        self._data_file_label = QLabel("No file selected")
        self._data_file_label.setWordWrap(True)
        data_file_btn = QPushButton("Browse...")
        data_file_btn.clicked.connect(self._browse_data_file)
        data_row = QHBoxLayout()
        data_row.addWidget(self._data_file_label, 1)
        data_row.addWidget(data_file_btn)
        file_layout.addRow("Data File:", data_row)

        self._submission_file_label = QLabel("None (start fresh)")
        self._submission_file_label.setWordWrap(True)
        submission_file_btn = QPushButton("Browse...")
        submission_file_btn.clicked.connect(self._browse_submission_file)
        submission_row = QHBoxLayout()
        submission_row.addWidget(self._submission_file_label, 1)
        submission_row.addWidget(submission_file_btn)
        file_layout.addRow("Initial Solution:", submission_row)

        layout.addWidget(file_group)

        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout(algo_group)

        self._algo_combo = QComboBox()
        for algo_cls in ALGORITHM_REGISTRY:
            self._algo_combo.addItem(algo_cls.name(), algo_cls)
        self._algo_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        algo_layout.addWidget(self._algo_combo)

        self._params_group = QGroupBox("Parameters")
        self._params_layout = QFormLayout(self._params_group)
        algo_layout.addWidget(self._params_group)

        layout.addWidget(algo_group)

        control_group = QGroupBox("Control")
        control_layout = QHBoxLayout(control_group)

        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self.run_requested.emit)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        self._stop_btn.setEnabled(False)

        control_layout.addWidget(self._run_btn)
        control_layout.addWidget(self._stop_btn)

        layout.addWidget(control_group)
        layout.addStretch()

        if ALGORITHM_REGISTRY:
            self._on_algorithm_changed(0)

    def _browse_data_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self._data_file_label.setText(file_path)
            self.data_file_changed.emit(file_path)

    def _browse_submission_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Initial Solution", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self._submission_file_label.setText(file_path)
            self.submission_file_changed.emit(file_path)

    def _on_algorithm_changed(self, index: int) -> None:
        algo_cls = self._algo_combo.itemData(index)
        if algo_cls is None:
            return

        self._current_algorithm = algo_cls
        self._rebuild_params(algo_cls.parameters())
        self.algorithm_changed.emit(algo_cls)

    def _rebuild_params(self, params: list[ParameterDef]) -> None:
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._param_widgets.clear()

        for param in params:
            widget = self._create_param_widget(param)
            self._param_widgets[param.name] = widget
            self._params_layout.addRow(f"{param.label}:", widget)

    def _create_param_widget(self, param: ParameterDef) -> QWidget:
        if param.param_type == int:
            widget = QSpinBox()
            widget.setRange(
                param.min_value if param.min_value is not None else 0,
                param.max_value if param.max_value is not None else 2_147_483_647
            )
            if param.step is not None:
                widget.setSingleStep(param.step)
            widget.setValue(param.default)
            widget.valueChanged.connect(self._emit_params_changed)
        elif param.param_type == float:
            widget = QDoubleSpinBox()
            widget.setRange(
                param.min_value if param.min_value is not None else 0.0,
                param.max_value if param.max_value is not None else 1e9
            )
            if param.step is not None:
                widget.setSingleStep(param.step)
            widget.setDecimals(4)
            widget.setValue(param.default)
            widget.valueChanged.connect(self._emit_params_changed)
        else:
            widget = QLabel(str(param.default))

        return widget

    def _emit_params_changed(self) -> None:
        self.parameters_changed.emit(self.get_parameters())

    def get_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                params[name] = widget.value()
        return params

    def get_algorithm_class(self) -> type[Algorithm] | None:
        return self._current_algorithm

    def get_data_file(self) -> str | None:
        text = self._data_file_label.text()
        if text == "No file selected":
            return None
        return text

    def get_submission_file(self) -> str | None:
        text = self._submission_file_label.text()
        if text == "None (start fresh)":
            return None
        return text

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._algo_combo.setEnabled(not running)
        for widget in self._param_widgets.values():
            widget.setEnabled(not running)
