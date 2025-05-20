"""Download and unzip from here to https://doi.org/10.24432/C5V60Q `uci_data`.

Then:
    - remove the leading text before the header form both the training and test data.
"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.concat(
    [
        pd.read_csv("uci_data/aps_failure_training_set.csv", na_values="na"),
        pd.read_csv("uci_data/aps_failure_test_set.csv", na_values="na"),
    ]
)

target_feature = "AirPressureSystemFailure"

data = data.rename(columns={"class": target_feature})

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    custom_data_split_for_shift=-16000,  # only the test data
    n_shift_repeats=1,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data.to_csv("APSFailure.csv", index=False)
