"""Download the preprocessed  data from https://www.agnostic.inf.ethz.ch/datasets/DataAgnos/HIVA.zip
and unzip it into a directory called "challenge_data" and download the raw data and
unzip it into "challenge_data_original".

For validation labels, you could use https://www.agnostic.inf.ethz.ch/datasets/ValidAgnos.zip
- but we do not have non-binarized validation labels, so we ignore them.
"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("challenge_data/hiva_train.data", sep=" ", header=None)
y_data = pd.read_csv("challenge_data_original/hiva_train.mlabels", header=None)
data.columns = [
    f"molecule_structure_property_{i + 1}" for i in range(len(data.columns))
]

# Remove trailing whitespace column
data = data.drop(columns=["molecule_structure_property_1618"])

target_feature_new = "CompoundActivity"
data[target_feature_new] = y_data[0]

data = data.astype("category")  # for proper checks.

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature_new,
    n_shift_repeats=1,  # takes very long due to many features.
)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("hiva_agnostic.csv", index=False)
