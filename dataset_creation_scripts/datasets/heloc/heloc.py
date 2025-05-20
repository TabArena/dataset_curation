"""Download the .csv from here https://www.kaggle.com/datasets/arunjangir245/healthcare-insurance-expenses/."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("heloc_dataset_v1 (1).csv")
target_feature = "RiskPerformance"

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("heloc.csv", index=False)
