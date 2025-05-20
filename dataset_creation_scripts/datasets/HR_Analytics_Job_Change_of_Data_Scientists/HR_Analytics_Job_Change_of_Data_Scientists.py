"""Download the train data from here https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists?select=aug_test.csv."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("aug_train.csv")
target_feature = "LookingForJobChange"
data = data.rename(columns={"target": target_feature})
data = data.drop(columns=["enrollee_id"])
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})


run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("HR_Analytics_Job_Change_of_Data_Scientists.csv", index=False)
