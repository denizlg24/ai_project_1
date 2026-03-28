import os
import pandas as pd


def load_data(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename, index_col='family_id')
    return data


def find_best_result(directory: str) -> tuple[str | None, float]:
    best_score = float('inf')
    best_file = None
    for filename in os.listdir(directory):
        if filename.startswith('submission_') and filename.endswith('.csv'):
            score = float(filename[len('submission_'):-len('.csv')])
            if score < best_score:
                best_score = score
                best_file = filename
    return best_file, best_score


def process_data(data: pd.DataFrame) -> tuple[dict[int, int], dict[str, dict[int, int]]]:
    family_size_dict = data[['n_people']].to_dict()['n_people']
    cols = [f'choice_{i}' for i in range(10)]
    choice_dict = data[cols].to_dict()
    return family_size_dict, choice_dict
