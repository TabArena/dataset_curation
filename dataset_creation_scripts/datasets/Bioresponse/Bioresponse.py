"""Download the train.csv from here https://www.kaggle.com/c/bioresponse/data?select=train.csv.
and put in this folder.
"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("train.csv")
target_feature = "MoleculeElicitsResponse"
data = data.rename(columns={"Activity": target_feature})
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    n_shift_repeats=1,
    skip_feature_importance_shift_detection=True,  # too slow, use other checks
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("Bioresponse.csv", index=False)
