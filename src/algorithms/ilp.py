import time
from typing import Any, Callable

import pulp
from core import Problem
from core.problem import MIN_OCCUPANCY, MAX_OCCUPANCY, N_DAYS
from .base import Algorithm, AlgorithmState, ParameterDef, ProgressData


class ILPSolver(Algorithm):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.max_choices = 10
        self.time_limit = 60
        self.mip_gap = 0.01
        # Linear surrogate weights for accounting-related behavior.
        self.occupancy_weight = 1.0
        self.smoothness_weight = 10.0

    @classmethod
    def name(cls) -> str:
        return "ILP Solver (PuLP)"

    @classmethod
    def parameters(cls) -> list[ParameterDef]:
        return [
            ParameterDef("max_choices", "Max Choices to Consider", int, 10, 1, 10, 1),
            ParameterDef("time_limit", "Time Limit (s)", int, 60, 10, 3600, 10),
            ParameterDef("mip_gap", "MIP Gap", float, 0.01, 0.0, 1.0, 0.001),
            ParameterDef("occupancy_weight", "Accounting Weight (Occupancy)", float, 1.0, 0.0, 1000.0, 0.1),
            ParameterDef("smoothness_weight", "Accounting Weight (Smoothness)", float, 10.0, 0.0, 1000.0, 0.1),
        ]

    def configure(self, **params: Any) -> None:
        self.max_choices = int(params.get("max_choices", self.max_choices))
        self.time_limit = int(params.get("time_limit", self.time_limit))
        self.mip_gap = float(params.get("mip_gap", self.mip_gap))
        self.occupancy_weight = float(params.get("occupancy_weight", self.occupancy_weight))
        self.smoothness_weight = float(params.get("smoothness_weight", self.smoothness_weight))

    def run(self, progress_callback: Callable[[ProgressData], None] | None = None) -> float:
        self.state = AlgorithmState.RUNNING
        self._stop_requested = False
        self._pause_requested = False
        self._start_timer()

        initial_score = self.problem.total_score()

        if progress_callback:
            progress_callback(ProgressData(
                iteration=0,
                current_score=initial_score,
                best_score=initial_score,
                elapsed_seconds=0,
                extra={"mip_status": 0.0},
            )) # 0: Building

        # Create the LP problem
        prob = pulp.LpProblem("SantaWorkshop", pulp.LpMinimize)

        # Decision variables: x[f, d] = 1 if family f is assigned to day d
        # We only consider the top N choices for each family to keep the problem small
        x = {}
        for fid in range(self.problem.num_families):
            for i in range(self.max_choices):
                day = self.problem.choices[fid][i]
                x[fid, day] = pulp.LpVariable(f"x_{fid}_{day}", 0, 1, pulp.LpBinary)

        # Objective (part 1): Minimize preference costs.
        obj_terms = []
        for fid in range(self.problem.num_families):
            choices = self.problem.choices[fid]
            size = self.problem.family_size[fid]
            for i in range(self.max_choices):
                day = choices[i]
                cost = Problem._preference_cost(size, choices, day)
                obj_terms.append(x[fid, day] * cost)

        # Occupancy variables per day (linked to assignments).
        occupancy: dict[int, pulp.LpVariable] = {
            day: pulp.LpVariable(f"occ_{day}", 0, MAX_OCCUPANCY, pulp.LpContinuous)
            for day in range(1, N_DAYS + 1)
        }

        # Linear surrogate for accounting behavior:
        # 1) Penalize occupancy above minimum occupancy.
        # 2) Penalize day-to-day occupancy changes.
        occupancy_excess: dict[int, pulp.LpVariable] = {
            day: pulp.LpVariable(f"occ_excess_{day}", 0, MAX_OCCUPANCY - MIN_OCCUPANCY, pulp.LpContinuous)
            for day in range(1, N_DAYS + 1)
        }
        occupancy_diff: dict[int, pulp.LpVariable] = {
            day: pulp.LpVariable(f"occ_diff_{day}", 0, MAX_OCCUPANCY - MIN_OCCUPANCY, pulp.LpContinuous)
            for day in range(1, N_DAYS)
        }

        accounting_terms = []
        for day in range(1, N_DAYS + 1):
            accounting_terms.append(self.occupancy_weight * occupancy_excess[day])
        for day in range(1, N_DAYS):
            accounting_terms.append(self.smoothness_weight * occupancy_diff[day])

        prob += pulp.lpSum(obj_terms) + pulp.lpSum(accounting_terms)

        # Constraint 1: Each family must be assigned to exactly one day
        for fid in range(self.problem.num_families):
            prob += pulp.lpSum([x[fid, day] for day in self.problem.choices[fid][:self.max_choices]]) == 1

        # Constraint 2: Daily occupancy must be within bounds [125, 300]
        for day in range(1, N_DAYS + 1):
            day_occupancy = []
            for fid in range(self.problem.num_families):
                if day in self.problem.choices[fid][:self.max_choices]:
                    day_occupancy.append(x[fid, day] * self.problem.family_size[fid])

            # Link occupancy variable to assignment decisions.
            prob += occupancy[day] == pulp.lpSum(day_occupancy)

            # Occupancy excess above the minimum occupancy.
            prob += occupancy_excess[day] >= occupancy[day] - MIN_OCCUPANCY
            
            if day_occupancy:
                prob += occupancy[day] >= MIN_OCCUPANCY
                prob += occupancy[day] <= MAX_OCCUPANCY
            else:
                # If no family can pick this day within their top N choices, the problem might be infeasible
                # but we'll let the solver find out.
                pass

        # Absolute day-to-day occupancy change linearization.
        for day in range(1, N_DAYS):
            prob += occupancy_diff[day] >= occupancy[day] - occupancy[day + 1]
            prob += occupancy_diff[day] >= occupancy[day + 1] - occupancy[day]

        # Solve the problem
        solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit, gapRel=self.mip_gap, msg=0)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != 'Optimal' and pulp.LpStatus[prob.status] != 'Feasible' :
             self.state = AlgorithmState.FINISHED
             return float('inf')

        # Update problem assignment
        new_assignment = self.problem.assignment[:]
        for fid in range(self.problem.num_families):
            for i in range(self.max_choices):
                day = self.problem.choices[fid][i]
                if pulp.value(x[fid, day]) > 0.5:
                    new_assignment[fid] = day
                    break

        # Calculate new daily occupancy and preference costs
        daily_occ = {d: 0 for d in range(1, N_DAYS + 1)}
        pref_costs = {}
        for fid, day in enumerate(new_assignment):
            daily_occ[day] += self.problem.family_size[fid]
            pref_costs[fid] = Problem._preference_cost(
                self.problem.family_size[fid], self.problem.choices[fid], day
            )

        # Update problem state
        self.problem.assignment = new_assignment
        self.problem.daily_occupancy = daily_occ
        self.problem.preference_costs = pref_costs

        final_score = self.problem.total_score()
        if progress_callback:
            progress_callback(ProgressData(
                iteration=1,
                current_score=final_score,
                best_score=final_score,
                elapsed_seconds=self._elapsed(),
                extra={"mip_status": 2.0},
            ))

        self.state = AlgorithmState.FINISHED
        return final_score
