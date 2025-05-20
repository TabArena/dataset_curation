"""The script assumes that you downloaded the data from the UCI repo and
unzipped it into directory called `uci_data` in the current working directory.
"""


from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/splice.data", index_col=False, header=None)
# Strip tab-like whitespaces from original data
data = data.map(lambda x: x.strip())
data = data.drop(columns=[1]) # Drop instance ID
target_feature = "SiteType"

split_data = data[2].apply(lambda x: pd.Series(list(x)))
# Generate new column names from -30 to +30 excluding 0
split_data.columns = [f"position_{i}" for i in range(-30, 31) if i != 0]
split_data[target_feature] = data[0]
data = split_data

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("splice.csv", index=False)

