"""Download and unzip from here https://doi.org/10.24432/C5F88Q to `uci_data/` folder."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/online_shoppers_intention.csv")
target_feature = "Revenue"

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    n_shift_repeats=1,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("online_shoppers_intention.csv", index=False)
