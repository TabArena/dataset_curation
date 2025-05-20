"""Download and unzip from here to https://doi.org/10.24432/C5K306 and
then unzip the bank.zip to `uci_data`.
"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/bank-full.csv", sep=";")
target_feature = "SubscribeTermDeposit"
data = data.rename(columns={"y": target_feature})

# From the original description: "last contact duration, in seconds (numeric). Important
# note: this attribute highly affects the output target (e.g., if duration=0
# then y='no'). Yet, the duration is not known before a call is performed.
# Also, after the end of the call y is obviously known. Thus, this input should only be
# included for benchmark purposes and should be discarded if the intention is to have a
# realistic predictive model."
# Thus: any information about the last contact is leaking the target feature and would
# not be known in a real world scenario.
# Therefore, we drop all features related to the last contact
data = data.drop(columns=["day", "month", "duration"])
# Note: "pdays" is not the last contact related to the current campaing and therefore would be
# available in a real world scenario.

# Data is ordered by time and thus has a shift based on the order by default.
# If we random shuffle the data, the shift is gone.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("bank-marketing.csv", index=False)
