import pulp
import random
from typing import Any, Callable

from core import Problem
from core.problem import MIN_OCCUPANCY, MAX_OCCUPANCY, N_DAYS
from .base import Algorithm, AlgorithmState, ParameterDef, ProgressData


class ILPLNSSolver(Algorithm):
    """
    LNS com ILP — desenhado para refinar uma boa solução inicial (ex: output do SA).

    Estratégia:
    - Se a solução for inviável, repara primeiro com greedy
    - Destrói janelas PEQUENAS de dias (5-10) para manter o ILP tratável
    - Nunca aceita solução pior (reverte sempre)
    - Selecciona janelas pelos dias com maior custo de preferência

    RECOMENDADO: usar com "Initial Solution" = output do SA/VNS.
    """

    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.window_size = 5
        self.time_limit_per_it = 15
        self.max_iterations = 50

    @classmethod
    def name(cls) -> str:
        return "ILP + LNS (Janela de Dias)"

    @classmethod
    def parameters(cls) -> list[ParameterDef]:
        return [
            ParameterDef("window_size",       "Dias por Janela", int,  5,  2,  20, 1),
            ParameterDef("max_iterations",    "Iterações",       int, 50,  1, 500000, 1),
            ParameterDef("time_limit_per_it", "Tempo/it (s)",    int, 15,  2,  180, 1),
        ]

    def configure(self, **params: Any) -> None:
        self.window_size       = int(params.get("window_size",       self.window_size))
        self.max_iterations    = int(params.get("max_iterations",    self.max_iterations))
        self.time_limit_per_it = int(params.get("time_limit_per_it", self.time_limit_per_it))

    # ------------------------------------------------------------------
    # Reparação greedy (só usada se solução for inviável)
    # ------------------------------------------------------------------

    def _is_feasible(self) -> bool:
        occ = self.problem.daily_occupancy
        return all(MIN_OCCUPANCY <= occ[d] <= MAX_OCCUPANCY for d in range(1, N_DAYS + 1))

    def _repair(self) -> None:
        for _ in range(200_000):
            occ = self.problem.daily_occupancy
            overflow  = [d for d in range(1, N_DAYS + 1) if occ[d] > MAX_OCCUPANCY]
            underflow = [d for d in range(1, N_DAYS + 1) if occ[d] < MIN_OCCUPANCY]
            if not overflow and not underflow:
                break

            if overflow:
                day_from = max(overflow, key=lambda d: occ[d])
                candidates = sorted(
                    [f for f, d in enumerate(self.problem.assignment) if d == day_from],
                    key=lambda f: self.problem.family_size[f], reverse=True
                )
                moved = False
                for fid in candidates:
                    for new_day in self.problem.choices[fid]:
                        if new_day != day_from and self.problem.is_feasible_move(fid, new_day):
                            _, new_pref = self.problem.delta_cost(fid, new_day)
                            self.problem.apply_move(fid, new_day, new_pref)
                            moved = True
                            break
                    if moved:
                        break
                if not moved:
                    for fid in candidates:
                        size = self.problem.family_size[fid]
                        for new_day in range(1, N_DAYS + 1):
                            if new_day != day_from and occ[new_day] + size <= MAX_OCCUPANCY:
                                self.problem.daily_occupancy[day_from] -= size
                                self.problem.daily_occupancy[new_day]  += size
                                self.problem.assignment[fid] = new_day
                                self.problem.preference_costs[fid] = Problem._preference_cost(
                                    size, self.problem.choices[fid], new_day)
                                moved = True
                                break
                        if moved:
                            break

            elif underflow:
                day_to = min(underflow, key=lambda d: occ[d])
                moved = False
                for day_from in sorted(
                    [d for d in range(1, N_DAYS + 1) if occ[d] > MIN_OCCUPANCY and d != day_to],
                    key=lambda d: occ[d], reverse=True
                ):
                    for fid in [f for f, d in enumerate(self.problem.assignment) if d == day_from]:
                        size = self.problem.family_size[fid]
                        if occ[day_to] + size <= MAX_OCCUPANCY and occ[day_from] - size >= 0:
                            self.problem.daily_occupancy[day_from] -= size
                            self.problem.daily_occupancy[day_to]   += size
                            self.problem.assignment[fid] = day_to
                            self.problem.preference_costs[fid] = Problem._preference_cost(
                                size, self.problem.choices[fid], day_to)
                            moved = True
                            break
                    if moved:
                        break
                if not moved:
                    break

    # ------------------------------------------------------------------
    # Selecção de janela guiada por custo de preferência
    # ------------------------------------------------------------------

    def _pick_window_start(self) -> int:
        if random.random() < 0.3:
            return random.randint(1, N_DAYS - self.window_size + 1)

        day_cost: dict[int, float] = {d: 0.0 for d in range(1, N_DAYS + 1)}
        for fid, day in enumerate(self.problem.assignment):
            day_cost[day] += self.problem.preference_costs.get(fid, 0.0)

        best_start = 1
        best_cost  = -1.0
        window_cost = sum(day_cost[d] for d in range(1, self.window_size + 1))
        for start in range(1, N_DAYS - self.window_size + 2):
            if window_cost > best_cost:
                best_cost  = window_cost
                best_start = start
            if start + self.window_size <= N_DAYS:
                window_cost += day_cost[start + self.window_size] - day_cost[start]
        return best_start

    # ------------------------------------------------------------------
    # Ponto de entrada
    # ------------------------------------------------------------------

    def run(self, progress_callback: Callable[[ProgressData], None] | None = None) -> float:
        self.state = AlgorithmState.RUNNING
        self._stop_requested = False
        self._pause_requested = False
        self._start_timer()

        # Fase 1: reparar se necessário
        if not self._is_feasible():
            self._repair()

        best_score    = self.problem.total_score()
        best_state    = self.problem.copy()
        current_score = best_score

        if progress_callback:
            progress_callback(ProgressData(
                iteration=0,
                current_score=current_score,
                best_score=best_score,
                elapsed_seconds=self._elapsed(),
                extra={"status": "Ready"},
            ))

        # Fase 2: LNS
        no_improve_count = 0

        for i in range(self.max_iterations):
            if self._should_stop():
                self.state = AlgorithmState.STOPPED
                break

            while self._pause_requested:
                self.state = AlgorithmState.PAUSED
                if self._should_stop():
                    break
            if self.state == AlgorithmState.PAUSED:
                self.state = AlgorithmState.RUNNING

            start_day   = self._pick_window_start()
            window_days = set(range(start_day, start_day + self.window_size))
            target_fids = [
                f for f, d in enumerate(self.problem.assignment)
                if d in window_days
            ]

            if not target_fids:
                continue

            snapshot = self.problem.copy()
            status   = self._solve_window(target_fids)
            new_score = self.problem.total_score()

            if status in ["Optimal", "Feasible"] and new_score < current_score:
                current_score    = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_state = self.problem.copy()
            else:
                self.problem.restore_from(snapshot)
                status = status + " (rev)"
                no_improve_count += 1

            if progress_callback:
                progress_callback(ProgressData(
                    iteration=i + 1,
                    current_score=current_score,
                    best_score=best_score,
                    elapsed_seconds=self._elapsed(),
                    extra={"status": status, "win": float(start_day)},
                ))

        self.problem.restore_from(best_state)
        if self.state != AlgorithmState.STOPPED:
            self.state = AlgorithmState.FINISHED
        return best_score

    # ------------------------------------------------------------------
    # Subproblema ILP
    # ------------------------------------------------------------------

    def _solve_window(self, target_fids: list[int]) -> str:
        target_set = set(target_fids)

        candidate_days: set[int] = set()
        for f in target_fids:
            candidate_days.update(self.problem.choices[f])

        fixed_occ: dict[int, int] = {d: 0 for d in range(1, N_DAYS + 1)}
        for f in range(self.problem.num_families):
            if f not in target_set:
                fixed_occ[self.problem.assignment[f]] += self.problem.family_size[f]

        # Só restringir dias onde fixed_occ não viola já por si só
        constrained_days = {d for d in candidate_days if fixed_occ[d] <= MAX_OCCUPANCY}

        prob = pulp.LpProblem("LNS_Window", pulp.LpMinimize)

        x = {
            (f, d): pulp.LpVariable(f"x_{f}_{d}", cat=pulp.LpBinary)
            for f in target_fids
            for d in self.problem.choices[f]
        }

        prob += pulp.lpSum(
            x[f, d] * Problem._preference_cost(
                self.problem.family_size[f], self.problem.choices[f], d
            )
            for f, d in x
        )

        for f in target_fids:
            prob += pulp.lpSum(x[f, d] for d in self.problem.choices[f]) == 1

        for d in constrained_days:
            day_load = pulp.lpSum(
                x[f, d] * self.problem.family_size[f]
                for f in target_fids if (f, d) in x
            )
            prob += day_load + fixed_occ[d] <= MAX_OCCUPANCY
            prob += day_load + fixed_occ[d] >= MIN_OCCUPANCY

        solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit_per_it, msg=0)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]

        if status in ["Optimal", "Feasible"]:
            for f in target_fids:
                for d in self.problem.choices[f]:
                    if (f, d) in x:
                        v = pulp.value(x[f, d])
                        if v is not None and v > 0.5:
                            self.problem.assignment[f] = d
                            break
            self._sync_metrics()

        return status

    # ------------------------------------------------------------------

    def _sync_metrics(self) -> None:
        daily_occ: dict[int, int] = {d: 0 for d in range(1, N_DAYS + 1)}
        pref_costs: dict[int, float] = {}
        for fid, day in enumerate(self.problem.assignment):
            size = self.problem.family_size[fid]
            daily_occ[day] += size
            pref_costs[fid] = Problem._preference_cost(
                size, self.problem.choices[fid], day
            )
        self.problem.daily_occupancy = daily_occ
        self.problem.preference_costs = pref_costs