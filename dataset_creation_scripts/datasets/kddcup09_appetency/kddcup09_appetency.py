"""From here https://kdd.org/kdd-cup/view/kdd-cup-2009/Data.

Download the following data and put it into this directory:
- orange_small_train.data.zip and unzip to orange_small_train_data
- orange_small_train_appetency.labels
"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("orange_small_train_data/orange_small_train.data", sep="\t")
y = pd.read_csv("orange_small_train_appetency.labels", header=None)
data["appetency"] = y[0]

empty_columns = [
    "Var8",
    "Var15",
    "Var20",
    "Var31",
    "Var32",
    "Var39",
    "Var42",
    "Var48",
    "Var52",
    "Var55",
    "Var79",
    "Var141",
    "Var167",
    "Var169",
    "Var175",
    "Var185",
    "Var209",
    "Var230",
]
data = data.drop(columns=empty_columns)


run_all_checks(
    data=data,
    classification=True,
    target_feature="appetency",
    n_shift_repeats=1,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("kddcup09_appetency.csv", index=False)
