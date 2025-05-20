"""Download the .csv from here https://www.kaggle.com/datasets/arunjangir245/healthcare-insurance-expenses/."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("insurance.csv")
target_feature = "charges"

run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("healthcare_insurance_expenses.csv", index=False)
