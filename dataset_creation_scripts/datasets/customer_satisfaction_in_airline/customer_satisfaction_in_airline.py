"""Download the .csv from here https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("Invistico_Airline.csv")
target_feature = "satisfaction"
data.columns = [
    c.replace(" ", "").replace("-", "").replace("/", "") for c in data.columns
]

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    n_shift_repeats=1,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("customer_satisfaction_in_airline.csv", index=False)
