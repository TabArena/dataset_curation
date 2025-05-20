"""Download the .csv from here https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("star_classification.csv")

target_feature = "ObjectType"
data = data.rename(columns={"class": target_feature})

data = data.drop_duplicates(subset=["obj_ID"])

# Note: the following is from a very naive domain knowledge perspective
# and should be checked with domain experts. But otherwise, the predictive task
# might leak. So we are better safe than sorry.
data = data.drop(
    columns=[
        "obj_ID",  # should not be predictive, also has duplicates?
        "spec_obj_ID",  # ID
        "run_ID",  # should not be predictive but might indicate confounding noise
        "rerun_ID",  # constant
        "field_ID",  # might indicate clusters/subgroups of data?
        "MJD",  # date of observation, should not be predictive?
    ]
)

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("SDSS17.csv", index=False)
