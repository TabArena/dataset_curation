"""Download the .csv from here https://www.kaggle.com/datasets/rajatkumar30/food-delivery-time."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("deliverytime.csv")

data = data.drop_duplicates(subset=["ID"])
data = data.drop(columns=["ID"])
target_feature = "Time_taken(min)"

run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("Food_Delivery_Time.csv", index=False)
