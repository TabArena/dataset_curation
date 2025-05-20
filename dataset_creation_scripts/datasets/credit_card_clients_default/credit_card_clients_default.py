"""Download and unzip from here https://doi.org/10.24432/C55S3H to `uci_data/` folder."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_excel("uci_data/default of credit card clients.xls", skiprows=[0])
target_feature = "DefaultOnPaymentNextMonth"
data = data.rename(columns={"default payment next month": target_feature})
data[target_feature] = data[target_feature].map({0: "No", 1: "Yes"})
data = data.drop(columns=["ID"])


# Data is ordered by chord-length, thus dist shift for original order
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("credit_card_clients_default.csv", index=False)
