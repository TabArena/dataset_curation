"""Download the train data from here https://www.kaggle.com/datasets/podsyp/is-this-a-good-customer."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("clients.csv")
target_feature = "bad_client_target"
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("Is-this-a-good-customer.csv", index=False)
