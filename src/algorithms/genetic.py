import random
from typing import Any, Callable

import numpy as np
from core import Problem
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
            ParameterDef("max_generations", "Max Generations", int, 1000, 100, 10000, 100),
        ]

    def configure(self, **params: Any) -> None:
        self.population_size = int(params.get("population_size", self.population_size))
        self.mutation_rate = float(params.get("mutation_rate", self.mutation_rate))
        self.max_generations = int(params.get("max_generations", self.max_generations))

    def _reproduce(self, parent1: Problem, parent2: Problem) -> Problem:
        child = parent1.copy()
        num_families = self.problem.num_families

        for i in range(num_families):
            if random.random() < 0.5:
                new_day = parent2.assignment[i]
                if child.is_feasible_move(i, new_day):
                    _, pref = child.delta_cost(i, new_day)
                    child.apply_move(i, new_day, pref)

        return child
    
    def _mutate(self, child: Problem) -> Problem:
        idx1 = random.randint(0, self.problem.num_families - 1)
        idx2 = random.randint(0, self.problem.num_families - 1)
    
        day1 = child.assignment[idx1]
        day2 = child.assignment[idx2]
        
        if child.is_feasible_move(idx1, day2) and child.is_feasible_move(idx2, day1):
            child.apply_move(idx1, day2, child.choices[idx1].index(day2) if day2 in child.choices[idx1] else 0)
            child.apply_move(idx2, day1, child.choices[idx2].index(day1) if day1 in child.choices[idx2] else 0)
        return child

    def run(self, progress_callback: Callable[[ProgressData], None] | None = None) -> float:
        self.state = AlgorithmState.RUNNING
        self._stop_requested = False
        self._pause_requested = False
        self._start_timer()

        population = []
        for _ in range(self.population_size):
            p = self.problem.copy()
            self._mutate(p)
            population.append(p)

        best_overall = self.problem.copy()
        best_score = best_overall.total_score()

        for gen in range(self.max_generations):
            if self._should_stop():
                self.state = AlgorithmState.STOPPED
                break

            scores = np.array([p.total_score() for p in population])

            shift_scores = scores - np.min(scores)
            exp_scores = np.exp(-shift_scores)
            fitness_probs = exp_scores / np.sum(exp_scores)

            new_population = []
            sorted_indices = np.argsort(scores)
            new_population.append(population[sorted_indices[0]].copy())
            new_population.append(population[sorted_indices[1]].copy())

            for _ in range(2,self.population_size):
                parent1, parent2 = random.choices(population, weights=fitness_probs, k=2)

                child = self._reproduce(parent1, parent2)

                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

            min_s = np.min(scores)
            if min_s < best_score:
                best_score = min_s
                best_overall = population[np.argmin(scores)].copy()

            if progress_callback:
                diversity = float(np.std(scores))
                progress_callback(ProgressData(
                    iteration=gen,
                    current_score=float(min_s),
                    best_score=float(best_score),
                    elapsed_seconds=self._elapsed(),
                    extra={"population_diversity": diversity},
                ))

        self.problem.restore_from(best_overall)
        if self.state != AlgorithmState.STOPPED:
            self.state = AlgorithmState.FINISHED
        return best_score