"""Download the .csv from here https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("Bank Customer Churn Prediction.csv")
target_feature = "churn"
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})
data = data.drop(columns=["customer_id"])

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("Bank_Customer_Churn.csv", index=False)
