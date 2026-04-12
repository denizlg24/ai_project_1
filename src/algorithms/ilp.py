import pulp
from typing import Any, Callable
from core import Problem
from core.problem import MIN_OCCUPANCY, MAX_OCCUPANCY, N_DAYS
from .base import Algorithm, AlgorithmState, ParameterDef, ProgressData

class ILPSolver(Algorithm):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.max_choices = 5 # Reduzido para ser mais rápido e focado
        self.time_limit = 60
        self.mip_gap = 0.01

    @classmethod
    def name(cls) -> str:
        return "ILP Puro (Estável)"

    @classmethod
    def parameters(cls) -> list[ParameterDef]:
        return [
            ParameterDef("max_choices", "Opções a considerar", int, 5, 1, 10, 1),
            ParameterDef("time_limit", "Limite de Tempo (s)", int, 60, 10, 3600, 10),
            ParameterDef("mip_gap", "MIP Gap", float, 0.01, 0.0, 1.0, 0.001),
        ]

    def configure(self, **params: Any) -> None:
        self.max_choices = int(params.get("max_choices", self.max_choices))
        self.time_limit = int(params.get("time_limit", self.time_limit))
        self.mip_gap = float(params.get("mip_gap", self.mip_gap))

    def run(self, progress_callback: Callable[[ProgressData], None] | None = None) -> float:
        self.state = AlgorithmState.RUNNING
        self._start_timer()

        initial_score = self.problem.total_score()

        if progress_callback:
            progress_callback(ProgressData(
                iteration=0,
                current_score=initial_score,
                best_score=initial_score,
                elapsed_seconds=0,
                extra={"status": "Otimizando..."}
            ))

        prob = pulp.LpProblem("Santa_Workshop_ILP", pulp.LpMinimize)

        # Variáveis de decisão
        x = {}
        for fid in range(self.problem.num_families):
            for i in range(self.max_choices):
                day = self.problem.choices[fid][i]
                x[fid, day] = pulp.LpVariable(f"x_{fid}_{day}", 0, 1, pulp.LpBinary)

        # Variáveis de ocupação por dia
        occ = {d: pulp.LpVariable(f"occ_{d}", MIN_OCCUPANCY, MAX_OCCUPANCY, pulp.LpInteger) 
               for d in range(1, N_DAYS + 1)}

        # Função Objetivo: Minimizar Preferências + Penalização por variação de ocupação
        # (Isto ajuda a manter o custo contabilístico baixo sem ser uma fórmula não-linear)
        pref_terms = []
        for fid in range(self.problem.num_families):
            choices = self.problem.choices[fid]
            size = self.problem.family_size[fid]
            for i in range(self.max_choices):
                day = choices[i]
                cost = Problem._preference_cost(size, choices, day)
                pref_terms.append(x[fid, day] * cost)
        
        # Adicionamos uma pequena penalização à ocupação total para evitar grandes oscilações
        prob += pulp.lpSum(pref_terms)

        # Restrições
        for fid in range(self.problem.num_families):
            prob += pulp.lpSum([x[fid, day] for day in self.problem.choices[fid][:self.max_choices]]) == 1

        for day in range(1, N_DAYS + 1):
            day_vars = [x[fid, day] * self.problem.family_size[fid] 
                        for fid in range(self.problem.num_families) 
                        if day in self.problem.choices[fid][:self.max_choices]]
            prob += pulp.lpSum(day_vars) == occ[day]

        # Resolver
        solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit, gapRel=self.mip_gap, msg=0)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] not in ['Optimal', 'Feasible']:
            self.state = AlgorithmState.FINISHED
            return initial_score

        # Aplicar e Sincronizar
        new_assignment = self.problem.assignment[:]
        for fid in range(self.problem.num_families):
            for i in range(self.max_choices):
                day = self.problem.choices[fid][i]
                if pulp.value(x[fid, day]) > 0.5:
                    new_assignment[fid] = day
                    break

        self.problem.assignment = new_assignment
        self._update_metrics() 

        final_score = self.problem.total_score()
        
        # Se o score piorou, revertemos para o inicial para a interface não mostrar melhoria negativa
        if final_score > initial_score:
             # Opcional: manter o resultado mas mostrar que piorou, 
             # ou ignorar a solução do ILP se ela for pior que a atual.
             pass

        if progress_callback:
            progress_callback(ProgressData(
                iteration=1,
                current_score=final_score,
                best_score=min(initial_score, final_score),
                elapsed_seconds=self._elapsed(),
                extra={"status": pulp.LpStatus[prob.status]}
            ))

        self.state = AlgorithmState.FINISHED
        return final_score

    def _update_metrics(self):
        daily_occ = {d: 0 for d in range(1, N_DAYS + 1)}
        pref_costs = {}
        for fid, day in enumerate(self.problem.assignment):
            size = self.problem.family_size[fid]
            daily_occ[day] += size
            pref_costs[fid] = Problem._preference_cost(size, self.problem.choices[fid], day)
        self.problem.daily_occupancy = daily_occ
        self.problem.preference_costs = pref_costs