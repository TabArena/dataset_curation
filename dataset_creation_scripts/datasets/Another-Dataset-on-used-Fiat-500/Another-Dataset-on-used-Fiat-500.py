"""Download the .csv from here https://www.kaggle.com/datasets/paolocons/another-fiat-500-dataset-1538-rows."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("automobile_dot_it_used_fiat_500_in_Italy_dataset_filtered.csv")
target_feature = "price"

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("Another-Dataset-on-used-Fiat-500.csv", index=False)
