"""Download the .csv from here https://www.kaggle.com/datasets/prachi13/customer-analytics"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("Train.csv")
target_feature = "ArrivedLate"
data = data.rename(columns={"Reached.on.Time_Y.N": target_feature})
data = data.drop(columns=["ID"])
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    n_shift_repeats=1,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("E-CommereShippingData.csv", index=False)
