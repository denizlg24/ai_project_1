from dataclasses import dataclass
import pandas as pd

from .data_loader import load_data, process_data


N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125


@dataclass
class Problem:
    family_size: dict[int, int]
    choices: dict[int, list[int]]
    assignment: list[int]
    daily_occupancy: dict[int, int]
    preference_costs: dict[int, float]

    @classmethod
    def from_files(cls, data_path: str, submission_path: str | None = None) -> "Problem":
        data = load_data(data_path)
        family_size, choice_dict = process_data(data)
        choices = {
            fid: [choice_dict[f'choice_{i}'][fid] for i in range(10)]
            for fid in family_size
        }

        if submission_path:
            submission = load_data(submission_path)
            assignment = submission['assigned_day'].tolist()
        else:
            assignment = [choices[fid][0] for fid in range(len(family_size))]

        daily_occupancy: dict[int, int] = {d: 0 for d in range(1, N_DAYS + 1)}
        for fid, day in enumerate(assignment):
            daily_occupancy[day] += family_size[fid]

        preference_costs = {
            fid: cls._preference_cost(family_size[fid], choices[fid], assignment[fid])
            for fid in family_size
        }

        return cls(
            family_size=family_size,
            choices=choices,
            assignment=assignment,
            daily_occupancy=daily_occupancy,
            preference_costs=preference_costs,
        )

    @staticmethod
    def _preference_cost(family_size: int, choice: list[int], assigned_day: int) -> float:
        n = family_size
        if assigned_day == choice[0]:
            return 0
        elif assigned_day == choice[1]:
            return 50
        elif assigned_day == choice[2]:
            return 50 + 9 * n
        elif assigned_day == choice[3]:
            return 100 + 9 * n
        elif assigned_day == choice[4]:
            return 200 + 9 * n
        elif assigned_day == choice[5]:
            return 200 + 18 * n
        elif assigned_day == choice[6]:
            return 300 + 18 * n
        elif assigned_day == choice[7]:
            return 300 + 36 * n
        elif assigned_day == choice[8]:
            return 400 + 36 * n
        elif assigned_day == choice[9]:
            return 500 + 36 * n + 199 * n
        else:
            return 500 + 36 * n + 398 * n

    def _accounting_cost_for_day(self, day: int) -> float:
        n_d = self.daily_occupancy[day]
        if n_d < MIN_OCCUPANCY or n_d > MAX_OCCUPANCY:
            return 1_000_000
        if day == N_DAYS:
            return max(0, (n_d - 125.0) / 400.0 * n_d ** 0.5)
        n_next = self.daily_occupancy[day + 1]
        diff = abs(n_d - n_next)
        return max(0, (n_d - 125.0) / 400.0 * n_d ** (0.5 + diff / 50.0))

    def total_score(self) -> float:
        pref_total = sum(self.preference_costs.values())
        acc_total = sum(self._accounting_cost_for_day(d) for d in range(1, N_DAYS + 1))
        return pref_total + acc_total

    def is_feasible_move(self, family_id: int, new_day: int) -> bool:
        n = self.family_size[family_id]
        old_day = self.assignment[family_id]
        if self.daily_occupancy[old_day] - n < MIN_OCCUPANCY:
            return False
        if self.daily_occupancy[new_day] + n > MAX_OCCUPANCY:
            return False
        return True

    def delta_cost(self, family_id: int, new_day: int) -> tuple[float, float]:
        old_day = self.assignment[family_id]
        n = self.family_size[family_id]

        old_pref = self.preference_costs[family_id]
        new_pref = self._preference_cost(n, self.choices[family_id], new_day)
        pref_delta = new_pref - old_pref

        affected = set()
        for d in (old_day, old_day - 1, new_day, new_day - 1):
            if 1 <= d <= N_DAYS:
                affected.add(d)

        acc_before = sum(self._accounting_cost_for_day(d) for d in affected)

        self.daily_occupancy[old_day] -= n
        self.daily_occupancy[new_day] += n

        acc_after = sum(self._accounting_cost_for_day(d) for d in affected)

        self.daily_occupancy[old_day] += n
        self.daily_occupancy[new_day] -= n

        return pref_delta + (acc_after - acc_before), new_pref

    def apply_move(self, family_id: int, new_day: int, new_pref: float) -> None:
        n = self.family_size[family_id]
        old_day = self.assignment[family_id]
        self.daily_occupancy[old_day] -= n
        self.daily_occupancy[new_day] += n
        self.assignment[family_id] = new_day
        self.preference_costs[family_id] = new_pref

    def copy(self) -> "Problem":
        return Problem(
            family_size=self.family_size,
            choices=self.choices,
            assignment=self.assignment[:],
            daily_occupancy=self.daily_occupancy.copy(),
            preference_costs=self.preference_costs.copy(),
        )

    def restore_from(self, other: "Problem") -> None:
        self.assignment[:] = other.assignment
        self.daily_occupancy.update(other.daily_occupancy)
        self.preference_costs.update(other.preference_costs)

    def to_submission(self, output_path: str) -> None:
        df = pd.DataFrame({
            'family_id': range(len(self.assignment)),
            'assigned_day': self.assignment
        })
        df.to_csv(output_path, index=False)

    @property
    def num_families(self) -> int:
        return len(self.family_size)
