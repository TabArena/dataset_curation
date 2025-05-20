"""Download the .csv from here https://www.kaggle.com/datasets/ddosad/datacamps-data-science-associate-certification."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("fitness_class_2212.csv")
data = data.drop(columns=["booking_id"])
target_feature = "attended"
data[target_feature] = abs(data[target_feature])
data[target_feature] = data[target_feature].map({0: "No", 1: "Yes"})

# Remove trailing text from column
data["days_before"] = data["days_before"].str.replace(" days", "").astype(int)
data["day_of_week"] = (
    data["day_of_week"]
    .str.replace("Wednesday", "Wed")
    .replace("Monday", "Mon")
    .replace("Fri.", "Fri")
)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("Fitness_Club.csv", index=False)
