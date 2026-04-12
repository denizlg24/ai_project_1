import pulp
from typing import Any, Callable
from core import Problem
from core.problem import MIN_OCCUPANCY, MAX_OCCUPANCY, N_DAYS
from .base import Algorithm, AlgorithmState, ParameterDef, ProgressData

class ILPLinearizedSolver(Algorithm):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.max_choices = 5
        self.time_limit = 60
        self.smooth_weight = 1.0 # Valor inicial mais seguro

    @classmethod
    def name(cls) -> str:
        return "ILP Linearizado (Smoothness)"

    @classmethod
    def parameters(cls) -> list[ParameterDef]:
        return [
            ParameterDef("max_choices", "Opções", int, 5, 1, 10, 1),
            ParameterDef("time_limit", "Tempo (s)", int, 60, 10, 3600, 10),
            ParameterDef("smooth_weight", "Peso Suavização", float, 1.0, 0.0, 100.0, 0.5),
        ]

    def configure(self, **params: Any) -> None:
        self.max_choices = int(params.get("max_choices", self.max_choices))
        self.time_limit = int(params.get("time_limit", self.time_limit))
        self.smooth_weight = float(params.get("smooth_weight", self.smooth_weight))

    def run(self, progress_callback: Callable[[ProgressData], None] | None = None) -> float:
        self.state = AlgorithmState.RUNNING
        self._start_timer()
        
        initial_score = self.problem.total_score()
        current_pref_total = sum(self.problem.preference_costs.values())
        
        if progress_callback:
            progress_callback(ProgressData(
                iteration=0,
                current_score=initial_score,
                best_score=initial_score,
                elapsed_seconds=0,
                extra={"status": "Otimizando..."}
            ))

        prob = pulp.LpProblem("ILP_Linearized", pulp.LpMinimize)
        x = {}
        for fid in range(self.problem.num_families):
            for i in range(self.max_choices):
                day = self.problem.choices[fid][i]
                x[fid, day] = pulp.LpVariable(f"x_{fid}_{day}", 0, 1, pulp.LpBinary)

        occ = {d: pulp.LpVariable(f"occ_{d}", MIN_OCCUPANCY, MAX_OCCUPANCY) for d in range(1, N_DAYS + 1)}
        diff = {d: pulp.LpVariable(f"diff_{d}", 0, MAX_OCCUPANCY) for d in range(1, N_DAYS)}

        pref_terms = []
        for fid in range(self.problem.num_families):
            for day in self.problem.choices[fid][:self.max_choices]:
                cost = Problem._preference_cost(self.problem.family_size[fid], self.problem.choices[fid], day)
                pref_terms.append(x[fid, day] * cost)
        
        prob += pulp.lpSum(pref_terms) + (self.smooth_weight * 0.5) * pulp.lpSum([diff[d] for d in range(1, N_DAYS)])

        # Restrição de Segurança: Não deixar as preferências piorarem mais de 2%
        prob += pulp.lpSum(pref_terms) <= current_pref_total * 1.02

        for fid in range(self.problem.num_families):
            prob += pulp.lpSum([x[fid, day] for day in self.problem.choices[fid][:self.max_choices]]) == 1

        for day in range(1, N_DAYS + 1):
            day_load = [x[fid, day] * self.problem.family_size[fid] for fid in range(self.problem.num_families) if (fid, day) in x]
            prob += pulp.lpSum(day_load) == occ[day]

        for d in range(1, N_DAYS):
            prob += diff[d] >= occ[d] - occ[d+1]
            prob += diff[d] >= occ[d+1] - occ[d]

        solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit, msg=0)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] in ['Optimal', 'Feasible']:
            self._apply_results(x)
        
        final_score = self.problem.total_score()
        if progress_callback:
            progress_callback(ProgressData(iteration=1, current_score=final_score, best_score=min(initial_score, final_score), 
                                         elapsed_seconds=self._elapsed(), extra={"status": pulp.LpStatus[prob.status]}))

        self.state = AlgorithmState.FINISHED
        return final_score

    def _apply_results(self, x):
        new_assignment = self.problem.assignment[:]
        for (fid, day), var in x.items():
            if pulp.value(var) > 0.5:
                new_assignment[fid] = day
        self.problem.assignment = new_assignment
        self._sync_metrics()

    def _sync_metrics(self):
        daily_occ = {d: 0 for d in range(1, N_DAYS + 1)}
        pref_costs = {}
        for fid, day in enumerate(self.problem.assignment):
            size = self.problem.family_size[fid]
            daily_occ[day] += size
            pref_costs[fid] = Problem._preference_cost(size, self.problem.choices[fid], day)
        self.problem.daily_occupancy = daily_occ
        self.problem.preference_costs = pref_costs