"""Download and unzip from here to https://doi.org/10.24432/C5GS39 `uci_data`."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/transfusion.data")
target_feature = "DonatedBloodInMarch2007"
data.columns = [
    "MonthsSinceLastDonation",
    "NumberOfDonations",
    "TotalBloodDonated",
    "MonthsSinceFirstDonation",
    target_feature
]
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})

# Shows a distribution shift, that means the original data was sorted
# by an order (by feature "MonthsSinceLastDonation"). Vanishes if we shuffle here.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("blood-transfusion-service-center.csv", index=False)
