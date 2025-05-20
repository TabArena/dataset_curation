"""Download and unzip from here https://doi.org/10.24432/C5GS4P to `uci_data/` folder."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/in-vehicle-coupon-recommendation.csv")
target_feature = "AcceptCoupon"
data = data.rename(columns={"Y": target_feature})
data[target_feature] = data[target_feature].map({0: "No", 1: "Yes"})

# Transform time to 24-hour format to be fully numeric.
data["time"] = (
    data["time"]
    .apply(
        lambda x: int(x.replace("AM", ""))
        if "AM" in x
        else int(x.replace("PM", "")) + 12
    )
    .astype(int)
)

# Drop constant columns
data = data.drop(columns=["toCoupon_GEQ5min"])
data = data.rename(columns={"passanger": "passenger"})

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("in_vehicle_coupon_recommendation.csv", index=False)
