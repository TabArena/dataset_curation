"""Download and unzip from https://doi.org/10.24432/C56S3T to `uci_data/`."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

red_data = pd.read_csv("uci_data/winequality-red.csv", sep=";")
white_data = pd.read_csv("uci_data/winequality-white.csv", sep=";")
red_data["wine_color"] = "red"
white_data["wine_color"] = "white"
data = pd.concat([red_data, white_data], ignore_index=True)
data.columns = data.columns.str.replace(" ", "_")
target_feature = "median_wine_quality"
data = data.rename(columns={"quality": target_feature})

# Requires shuffle as it has an obvious distribution shift due to concatenation of red and white wines.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("wine_quality.csv", index=False)

