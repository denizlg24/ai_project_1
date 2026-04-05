import math
import random
from typing import Any, Callable

from core import Problem
from .base import Algorithm, AlgorithmState, ParameterDef, ProgressData


class SimulatedAnnealing(Algorithm):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.max_iterations = 1_000_000
        self.calibration_samples = 1000
        self.target_acceptance = 0.25
        self.t_initial: float | None = None
        self.alpha: float | None = None

    @classmethod
    def name(cls) -> str:
        return "Simulated Annealing"

    @classmethod
    def parameters(cls) -> list[ParameterDef]:
        return [
            ParameterDef(
                name="max_iterations",
                label="Max Iterations",
                param_type=int,
                default=1_000_000,
                min_value=1000,
                max_value=100_000_000,
                step=100_000,
            ),
            ParameterDef(
                name="calibration_samples",
                label="Calibration Samples",
                param_type=int,
                default=1000,
                min_value=100,
                max_value=10_000,
                step=100,
            ),
            ParameterDef(
                name="target_acceptance",
                label="Target Acceptance",
                param_type=float,
                default=0.25,
                min_value=0.01,
                max_value=0.99,
                step=0.01,
            ),
        ]

    def configure(self, **params: Any) -> None:
        if "max_iterations" in params:
            self.max_iterations = int(params["max_iterations"])
        if "calibration_samples" in params:
            self.calibration_samples = int(params["calibration_samples"])
        if "target_acceptance" in params:
            self.target_acceptance = float(params["target_acceptance"])

    def _calibrate_temperature(self) -> float:
        """Sample feasible uphill cost deltas and set T so a typical (quantile) delta is accepted with probability ≈ target_acceptance via exp(-Δ/T)."""
        uphill_deltas: list[float] = []
        attempts = 0
        max_attempts = self.calibration_samples * 100

        while len(uphill_deltas) < self.calibration_samples and attempts < max_attempts:
            attempts += 1
            family_id = random.randint(0, self.problem.num_families - 1)
            new_day = random.choice(self.problem.choices[family_id])

            if new_day == self.problem.assignment[family_id]:
                continue
            if not self.problem.is_feasible_move(family_id, new_day):
                continue

            delta, _ = self.problem.delta_cost(family_id, new_day)
            if delta > 0:
                uphill_deltas.append(delta)

        if not uphill_deltas:
            return 1000.0

        uphill_deltas.sort()
        idx = int(len(uphill_deltas) * self.target_acceptance)
        acceptance_quantile = uphill_deltas[idx]
        return -acceptance_quantile / math.log(self.target_acceptance)

    def run(self, progress_callback: Callable[[ProgressData], None] | None = None) -> float:
        self.state = AlgorithmState.RUNNING
        self._stop_requested = False
        self._pause_requested = False
        self._start_timer()

        self.t_initial = self._calibrate_temperature()
        self.alpha = 0.01 ** (1.0 / self.max_iterations)

        t = self.t_initial
        current_score = self.problem.total_score()
        best_score = current_score
        best_state = self.problem.copy()

        report_interval = max(1, self.max_iterations // 1000)

        for iteration in range(self.max_iterations):
            if self._should_stop():
                self.state = AlgorithmState.STOPPED
                break

            while self._pause_requested:
                self.state = AlgorithmState.PAUSED
                if self._should_stop():
                    break

            if self.state == AlgorithmState.PAUSED:
                self.state = AlgorithmState.RUNNING

            family_id = random.randint(0, self.problem.num_families - 1)
            new_day = random.choice(self.problem.choices[family_id])

            if new_day == self.problem.assignment[family_id]:
                t *= self.alpha
                continue

            if not self.problem.is_feasible_move(family_id, new_day):
                t *= self.alpha
                continue

            delta, new_pref = self.problem.delta_cost(family_id, new_day)

            if delta < 0 or random.random() < math.exp(-delta / t):
                self.problem.apply_move(family_id, new_day, new_pref)
                current_score += delta

                if current_score < best_score:
                    best_score = current_score
                    best_state = self.problem.copy()

            t *= self.alpha

            if progress_callback and iteration % report_interval == 0:
                progress_callback(ProgressData(
                    iteration=iteration,
                    current_score=current_score,
                    best_score=best_score,
                    elapsed_seconds=self._elapsed(),
                    extra={"temperature": t},
                ))

        self.problem.restore_from(best_state)

        if self.state != AlgorithmState.STOPPED:
            self.state = AlgorithmState.FINISHED

        if progress_callback:
            progress_callback(ProgressData(
                iteration=self.max_iterations,
                current_score=best_score,
                best_score=best_score,
                elapsed_seconds=self._elapsed(),
                extra={"temperature": t},
            ))

        return best_score
