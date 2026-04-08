import os
from core.problem import Problem

# Evaluate sample submission
problem = Problem.from_files(
    data_path="../input/family_data.csv",
    submission_path="../output/sample_submission.csv"
)

score = problem.total_score()
print(f"Sample submission score: {score}")

# Rename the file
old_path = "../output/sample_submission.csv"
new_path = f"../output/sample_submission_score_{score}.csv"
os.rename(old_path, new_path)
print(f"Renamed to: {new_path}")
