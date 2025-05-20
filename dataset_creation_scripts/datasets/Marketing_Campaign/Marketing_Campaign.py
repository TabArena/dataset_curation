"""Download the .csv from here https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign/data
and rename it to input_marketing_campaign.csv"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("input_marketing_campaign.csv", sep=";")
target_feature = "Response"
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})
data = data.drop(columns=["ID", "Z_CostContact", "Z_Revenue"])

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("Marketing_Campaign.csv", index=False)
