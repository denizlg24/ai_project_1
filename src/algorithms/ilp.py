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

    @classmethod
    def name(cls) -> str:
        return "ILP Solver (PuLP)"

    @classmethod
    def parameters(cls) -> list[ParameterDef]:
        return [
            ParameterDef("max_choices", "Max Choices to Consider", int, 10, 1, 10, 1),
            ParameterDef("time_limit", "Time Limit (s)", int, 60, 10, 3600, 10),
            ParameterDef("mip_gap", "MIP Gap", float, 0.01, 0.0, 1.0, 0.001),
        ]

    def configure(self, **params: Any) -> None:
        self.max_choices = int(params.get("max_choices", self.max_choices))
        self.time_limit = int(params.get("time_limit", self.time_limit))
        self.mip_gap = float(params.get("mip_gap", self.mip_gap))

    def run(self, progress_callback: Callable[[ProgressData], None] | None = None) -> float:
        self.state = AlgorithmState.RUNNING
        self._stop_requested = False
        self._pause_requested = False
        self._start_timer()

        if progress_callback:
            progress_callback(ProgressData(0, 0, 0, 0, {"mip_status": 0.0})) # 0: Building

        # Create the LP problem
        prob = pulp.LpProblem("SantaWorkshop", pulp.LpMinimize)

        # Decision variables: x[f, d] = 1 if family f is assigned to day d
        # We only consider the top N choices for each family to keep the problem small
        x = {}
        for fid in range(self.problem.num_families):
            for i in range(self.max_choices):
                day = self.problem.choices[fid][i]
                x[fid, day] = pulp.LpVariable(f"x_{fid}_{day}", 0, 1, pulp.LpBinary)

        # Objective: Minimizing preference costs
        obj_terms = []
        for fid in range(self.problem.num_families):
            choices = self.problem.choices[fid]
            size = self.problem.family_size[fid]
            for i in range(self.max_choices):
                day = choices[i]
                cost = Problem._preference_cost(size, choices, day)
                obj_terms.append(x[fid, day] * cost)
        
        prob += pulp.lpSum(obj_terms)

        # Constraint 1: Each family must be assigned to exactly one day
        for fid in range(self.problem.num_families):
            prob += pulp.lpSum([x[fid, day] for day in self.problem.choices[fid][:self.max_choices]]) == 1

        # Constraint 2: Daily occupancy must be within bounds [125, 300]
        for day in range(1, N_DAYS + 1):
            day_occupancy = []
            for fid in range(self.problem.num_families):
                if day in self.problem.choices[fid][:self.max_choices]:
                    day_occupancy.append(x[fid, day] * self.problem.family_size[fid])
            
            if day_occupancy:
                prob += pulp.lpSum(day_occupancy) >= MIN_OCCUPANCY
                prob += pulp.lpSum(day_occupancy) <= MAX_OCCUPANCY
            else:
                # If no family can pick this day within their top N choices, the problem might be infeasible
                # but we'll let the solver find out.
                pass

        if progress_callback:
            progress_callback(ProgressData(0, 0, 0, self._elapsed(), {"mip_status": 1.0})) # 1: Solving

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
