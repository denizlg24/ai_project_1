import random
from typing import Any, Callable

from core import Problem
from core.problem import MIN_OCCUPANCY, MAX_OCCUPANCY
from .base import Algorithm, AlgorithmState, ParameterDef, ProgressData


class VariableNeighbourhoodSearch(Algorithm):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.max_iterations = 5_000
        self.local_search_max_iter = 50_000
        self.k_max = 4

    @classmethod
    def name(cls) -> str:
        return "Variable Neighbourhood Search"

    @classmethod
    def parameters(cls) -> list[ParameterDef]:
        return [
            ParameterDef(
                name="max_iterations",
                label="Max Iterations",
                param_type=int,
                default=5_000,
                min_value=100,
                max_value=100_000,
                step=500,
            ),
            ParameterDef(
                name="local_search_max_iter",
                label="Local Search Max Iter",
                param_type=int,
                default=50_000,
                min_value=1_000,
                max_value=500_000,
                step=5_000,
            ),
            ParameterDef(
                name="k_max",
                label="Neighbourhood Count",
                param_type=int,
                default=4,
                min_value=1,
                max_value=4,
                step=1,
            ),
        ]

    def configure(self, **params: Any) -> None:
        if "max_iterations" in params:
            self.max_iterations = int(params["max_iterations"])
        if "local_search_max_iter" in params:
            self.local_search_max_iter = int(params["local_search_max_iter"])
        if "k_max" in params:
            self.k_max = int(params["k_max"])

    def _shake_single_move(self, problem: Problem) -> bool:
        """Move a random family to a random preferred day."""
        for _ in range(100):
            fid = random.randint(0, problem.num_families - 1)
            new_day = random.choice(problem.choices[fid])
            if new_day == problem.assignment[fid]:
                continue
            if not problem.is_feasible_move(fid, new_day):
                continue
            _, new_pref = problem.delta_cost(fid, new_day)
            problem.apply_move(fid, new_day, new_pref)
            return True
        return False

    def _shake_swap(self, problem: Problem) -> bool:
        """Swap the assigned days of two random families."""
        for _ in range(100):
            fid1 = random.randint(0, problem.num_families - 1)
            fid2 = random.randint(0, problem.num_families - 1)
            if fid1 == fid2:
                continue
            day1 = problem.assignment[fid1]
            day2 = problem.assignment[fid2]
            if day1 == day2:
                continue
            n1 = problem.family_size[fid1]
            n2 = problem.family_size[fid2]
            new_occ1 = problem.daily_occupancy[day1] - n1 + n2
            new_occ2 = problem.daily_occupancy[day2] - n2 + n1
            if not (MIN_OCCUPANCY <= new_occ1 <= MAX_OCCUPANCY
                    and MIN_OCCUPANCY <= new_occ2 <= MAX_OCCUPANCY):
                continue
            problem.daily_occupancy[day1] = new_occ1
            problem.daily_occupancy[day2] = new_occ2
            problem.assignment[fid1] = day2
            problem.assignment[fid2] = day1
            problem.preference_costs[fid1] = Problem._preference_cost(
                n1, problem.choices[fid1], day2
            )
            problem.preference_costs[fid2] = Problem._preference_cost(
                n2, problem.choices[fid2], day1
            )
            return True
        return False

    def _shake_chain(self, problem: Problem) -> bool:
        """Move family A to B's day, then move B to a new preferred day."""
        for _ in range(200):
            fid_a = random.randint(0, problem.num_families - 1)
            fid_b = random.randint(0, problem.num_families - 1)
            if fid_a == fid_b:
                continue
            day_a = problem.assignment[fid_a]
            day_b = problem.assignment[fid_b]
            if day_a == day_b:
                continue

            if not problem.is_feasible_move(fid_a, day_b):
                continue
            _, pref_a = problem.delta_cost(fid_a, day_b)
            problem.apply_move(fid_a, day_b, pref_a)

            moved_b = False
            choices_b = list(problem.choices[fid_b])
            random.shuffle(choices_b)
            for new_day_b in choices_b:
                if new_day_b == problem.assignment[fid_b]:
                    continue
                if not problem.is_feasible_move(fid_b, new_day_b):
                    continue
                _, pref_b = problem.delta_cost(fid_b, new_day_b)
                problem.apply_move(fid_b, new_day_b, pref_b)
                moved_b = True
                break

            if not moved_b:
                undo_pref = Problem._preference_cost(
                    problem.family_size[fid_a], problem.choices[fid_a], day_a
                )
                problem.apply_move(fid_a, day_a, undo_pref)
                continue
            return True
        return False

    def _shake_day_redistribute(self, problem: Problem) -> bool:
        """Pick a random day, redistribute up to 5 families to other preferred days."""
        day = random.choice(list(problem.daily_occupancy.keys()))
        families_on_day = [
            fid for fid, d in enumerate(problem.assignment) if d == day
        ]
        if not families_on_day:
            return False
        random.shuffle(families_on_day)
        moved_any = False
        count = 0
        for fid in families_on_day:
            if count >= 5:
                break
            choices = list(problem.choices[fid])
            random.shuffle(choices)
            for new_day in choices:
                if new_day == day:
                    continue
                if not problem.is_feasible_move(fid, new_day):
                    continue
                _, new_pref = problem.delta_cost(fid, new_day)
                problem.apply_move(fid, new_day, new_pref)
                moved_any = True
                count += 1
                break
        return moved_any

    def _local_search(self, problem: Problem, score: float) -> float:
        """First-improvement local search using single-family moves (N1)."""
        improved = True
        iterations = 0

        while improved and iterations < self.local_search_max_iter:
            if self._should_stop():
                break
            improved = False
            families = list(range(problem.num_families))
            random.shuffle(families)

            for fid in families:
                if self._should_stop():
                    break
                iterations += 1
                if iterations >= self.local_search_max_iter:
                    break

                for new_day in problem.choices[fid]:
                    if new_day == problem.assignment[fid]:
                        continue
                    if not problem.is_feasible_move(fid, new_day):
                        continue
                    delta, new_pref = problem.delta_cost(fid, new_day)
                    if delta < 0:
                        problem.apply_move(fid, new_day, new_pref)
                        score += delta
                        improved = True
                        break

        return score

    def run(self, progress_callback: Callable[[ProgressData], None] | None = None) -> float:
        self.state = AlgorithmState.RUNNING
        self._stop_requested = False
        self._pause_requested = False
        self._start_timer()

        shakers = [
            self._shake_single_move,
            self._shake_swap,
            self._shake_chain,
            self._shake_day_redistribute,
        ]

        current_score = self.problem.total_score()
        best_score = current_score
        best_state = self.problem.copy()
        report_interval = max(1, self.max_iterations // 1000)

        k = 0
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

            pre_shake_score = current_score
            pre_shake = self.problem.copy()

            shake_fn = shakers[k % self.k_max]
            if not shake_fn(self.problem):
                k = (k + 1) % self.k_max
                continue

            shaken_score = self.problem.total_score()
            ls_score = self._local_search(self.problem, shaken_score)

            if ls_score < pre_shake_score:
                current_score = ls_score
                k = 0
                if ls_score < best_score:
                    best_score = ls_score
                    best_state = self.problem.copy()
            else:
                self.problem.restore_from(pre_shake)
                current_score = pre_shake_score
                k = (k + 1) % self.k_max

            if progress_callback and iteration % report_interval == 0:
                progress_callback(ProgressData(
                    iteration=iteration,
                    current_score=current_score,
                    best_score=best_score,
                    elapsed_seconds=self._elapsed(),
                    extra={"neighbourhood": float(k + 1)},
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
                extra={"neighbourhood": float(k + 1)},
            ))

        return best_score
