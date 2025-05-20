"""The script assumes that you downloaded the .csv file from Kaggle and named it "input_diabetes.csv".
"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("input_diabetes.csv")

target_feature_new = "TestedPositiveForDiabetes"
data = data.rename(columns={"Outcome": target_feature_new})
data[target_feature_new] = data[target_feature_new].replace(1, "Yes").replace(0, "No")

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature_new,
)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("diabetes.csv", index=False)
