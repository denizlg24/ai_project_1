import random
from typing import Any, Callable

import numpy as np
from core import Problem
from core.problem import MAX_OCCUPANCY, MIN_OCCUPANCY
from .base import Algorithm, AlgorithmState, ParameterDef, ProgressData


class GeneticAlgorithm(Algorithm):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.population_size = 100
        self.mutation_rate = 0.05
        self.max_generations = 1000

    @classmethod
    def name(cls) -> str:
        return "Genetic Algorithm"

    @classmethod
    def parameters(cls) -> list[ParameterDef]:
        return [
            ParameterDef("population_size", "Population Size", int, 100, 10, 500, 10),
            ParameterDef("mutation_rate", "Mutation Rate", float, 0.05, 0.0, 1.0, 0.01),
            ParameterDef(
                "max_generations", "Max Generations", int, 1000, 100, 10000, 100
            ),
        ]

    def configure(self, **params: Any) -> None:
        self.population_size = int(params.get("population_size", self.population_size))
        self.mutation_rate = float(params.get("mutation_rate", self.mutation_rate))
        self.max_generations = int(params.get("max_generations", self.max_generations))

    def _reproduce(self, parent1: Problem, parent2: Problem) -> Problem:
        child = parent1.copy()
        num_families = child.num_families

        for i in range(num_families):
            if random.random() < 0.5:
                new_day = parent2.assignment[i]
                if new_day == child.assignment[i]:
                    continue
                if child.is_feasible_move(i, new_day):
                    _, pref = child.delta_cost(i, new_day)
                    child.apply_move(i, new_day, pref)

        return child

    def _mutate(self, child: Problem) -> Problem:
        for _ in range(30):
            idx1 = random.randint(0, child.num_families - 1)
            idx2 = random.randint(0, child.num_families - 1)
            if idx1 == idx2:
                continue

            day1 = child.assignment[idx1]
            day2 = child.assignment[idx2]
            if day1 == day2:
                continue

            n1 = child.family_size[idx1]
            n2 = child.family_size[idx2]
            new_occ1 = child.daily_occupancy[day1] - n1 + n2
            new_occ2 = child.daily_occupancy[day2] - n2 + n1

            if not (
                MIN_OCCUPANCY <= new_occ1 <= MAX_OCCUPANCY
                and MIN_OCCUPANCY <= new_occ2 <= MAX_OCCUPANCY
            ):
                continue

            child.daily_occupancy[day1] = new_occ1
            child.daily_occupancy[day2] = new_occ2
            child.assignment[idx1] = day2
            child.assignment[idx2] = day1
            child.preference_costs[idx1] = Problem._preference_cost(
                child.family_size[idx1], child.choices[idx1], day2
            )
            child.preference_costs[idx2] = Problem._preference_cost(
                child.family_size[idx2], child.choices[idx2], day1
            )
            return child

        for _ in range(30):
            fid = random.randint(0, child.num_families - 1)
            choices = list(child.choices[fid])
            random.shuffle(choices)
            for new_day in choices:
                if new_day == child.assignment[fid]:
                    continue
                if not child.is_feasible_move(fid, new_day):
                    continue
                _, new_pref = child.delta_cost(fid, new_day)
                child.apply_move(fid, new_day, new_pref)
                return child

        return child

    def run(
        self, progress_callback: Callable[[ProgressData], None] | None = None
    ) -> float:
        self.state = AlgorithmState.RUNNING
        self._stop_requested = False
        self._pause_requested = False
        self._start_timer()

        population = []
        for _ in range(self.population_size):
            p = self.problem.copy()
            for _ in range(random.randint(1, 3)):
                self._mutate(p)
            population.append(p)

        best_overall = self.problem.copy()
        best_score = best_overall.total_score()

        for gen in range(self.max_generations):
            if self._should_stop():
                self.state = AlgorithmState.STOPPED
                break

            while self._pause_requested:
                self.state = AlgorithmState.PAUSED
                if self._should_stop():
                    break

            if self.state == AlgorithmState.PAUSED:
                self.state = AlgorithmState.RUNNING

            scores = np.array([p.total_score() for p in population], dtype=float)
            best_idx = int(np.argmin(scores))
            min_s = float(scores[best_idx])
            if min_s < best_score:
                best_score = min_s
                best_overall = population[best_idx].copy()

            shift_scores = scores - np.min(scores)
            exp_scores = np.exp(-shift_scores)
            sum_fitness = float(np.sum(exp_scores))
            if sum_fitness <= 0 or not np.isfinite(sum_fitness):
                fitness_probs = np.full(
                    self.population_size, 1.0 / self.population_size
                )
            else:
                fitness_probs = exp_scores / sum_fitness

            new_population = []
            sorted_indices = np.argsort(scores)
            new_population.append(population[sorted_indices[0]].copy())
            new_population.append(population[sorted_indices[1]].copy())

            for _ in range(2, self.population_size):
                parent1, parent2 = random.choices(
                    population, weights=fitness_probs, k=2
                )

                child = self._reproduce(parent1, parent2)

                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

            if progress_callback:
                diversity = float(np.std(scores))
                progress_callback(
                    ProgressData(
                        iteration=gen,
                        current_score=min_s,
                        best_score=best_score,
                        elapsed_seconds=self._elapsed(),
                        extra={"population_diversity": diversity},
                    )
                )

        self.problem.restore_from(best_overall)
        if self.state != AlgorithmState.STOPPED:
            self.state = AlgorithmState.FINISHED
        return best_score
