"""Download and unzip from here https://doi.org/10.24432/C5W902 to `uci_data/` folder."""

from __future__ import annotations

import arff
import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

with open("uci_data/seismic-bumps.arff") as f:
    data = arff.load(f)

data = pd.DataFrame(data["data"], columns=[x[0] for x in data["attributes"]])
target_feature = "HighEnergySeismicBump"
data = data.rename(columns={"class": target_feature})
data[target_feature] = data[target_feature].map({"1": "Yes", "0": "No"})

# Drop constant columns
data = data.loc[:, (data != data.iloc[0]).any()]

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("seismic-bumps.csv", index=False)
