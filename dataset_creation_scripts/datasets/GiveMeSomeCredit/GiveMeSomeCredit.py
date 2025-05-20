"""Download the train .csv from here https://www.kaggle.com/competitions/GiveMeSomeCredit/data."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("cs-training.csv", index_col=0, na_values="NA")
target_feature = "FinancialDistressNextTwoYears"
data = data.rename(columns={"SeriousDlqin2yrs": target_feature})
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("GiveMeSomeCredit.csv", index=False)
