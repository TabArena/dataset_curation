"""We get the source from OpenML (see below)."""

from __future__ import annotations

import openml
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = openml.datasets.get_dataset(1053).get_data()[0]

target_feature = "defects"

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("jm1.csv", index=False)
