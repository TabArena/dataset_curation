"""Download and unzip from here https://doi.org/10.24432/C5JG7B to `uci_data/` folder."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/qsar_fish_toxicity.csv", sep=";")
target_feature = "LC50"
data.columns = [
    "CIC0",
    "SM1_Dz(Z)",
    "GATS1i",
    "NdsCH",
    "NdssC",
    "MLOGP",
    target_feature,
]

# Original data is ordered and thus we have shift that vanishes after shuffling
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("QSAR_fish_toxicity.csv", index=False)
