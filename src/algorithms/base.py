import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from core import Problem


class AlgorithmState(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    FINISHED = auto()


@dataclass
class ParameterDef:
    name: str
    label: str
    param_type: type
    default: Any
    min_value: Any = None
    max_value: Any = None
    step: Any = None


@dataclass
class ProgressData:
    iteration: int
    current_score: float
    best_score: float
    elapsed_seconds: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


class Algorithm(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem
        self.state = AlgorithmState.IDLE
        self._stop_requested = False
        self._pause_requested = False
        self.max_time_seconds: float = 0

    def _start_timer(self) -> None:
        self._start_time = time.monotonic()

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    def _time_exceeded(self) -> bool:
        return self.max_time_seconds > 0 and self._elapsed() > self.max_time_seconds

    def _should_stop(self) -> bool:
        return self._stop_requested or self._time_exceeded()

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def parameters(cls) -> list[ParameterDef]:
        pass

    @abstractmethod
    def configure(self, **params: Any) -> None:
        pass

    @abstractmethod
    def run(self, progress_callback: Any | None = None) -> float:
        pass

    def request_stop(self) -> None:
        self._stop_requested = True

    def request_pause(self) -> None:
        self._pause_requested = True

    def resume(self) -> None:
        self._pause_requested = False

    def reset(self) -> None:
        self.state = AlgorithmState.IDLE
        self._stop_requested = False
        self._pause_requested = False
