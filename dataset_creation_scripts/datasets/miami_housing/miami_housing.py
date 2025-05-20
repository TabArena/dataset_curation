"""Download the arff file from https://www.openml.org/search?type=data&id=43093 and put
it into this folder.
"""

from __future__ import annotations

import arff
import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

with open("miami2016.arff") as f:
    data = arff.load(f)
data = pd.DataFrame(data["data"], columns=[x[0] for x in data["attributes"]])
target_feature = "SALE_PRC"

data = data.drop_duplicates(subset=["PARCELNO"])
data = data.drop(columns=["PARCELNO"])

# Original data is ordered and thus we have shift that vanishes after shuffling
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("miami_housing.csv", index=False)
