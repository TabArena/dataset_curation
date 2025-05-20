"""Download the .arff `dataset_` from https://www.openml.org/search?type=data&status=active&id=45538."""

from __future__ import annotations

import arff
import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

with open("dataset_") as f:
    data = arff.load(f)
data = pd.DataFrame(data["data"], columns=[x[0] for x in data["attributes"]])
target_feature = "Contaminated"
data = data.rename(columns={"class": target_feature})
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("hazelnut-spread-contaminant-detection.csv", index=False)
