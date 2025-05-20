"""Download and churn.tsv.gz from https://github.com/EpistasisLab/pmlb/tree/master/datasets/churn
and extract the churn.tsv and put it in this folder.
"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("churn.tsv", sep="\t")
target_feature = "CustomerChurned"
data = data.rename(columns={"target": target_feature})
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})
data.columns = data.columns.str.replace(" ", "_")
data = data.drop(columns=["phone_number"])

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("churn.csv", index=False)
