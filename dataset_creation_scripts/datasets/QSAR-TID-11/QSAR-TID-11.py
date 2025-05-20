"""Download the data from here https://www.openml.org/search?type=data&sort=runs&status=active&id=3050"""
from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks
import arff

with open('11.arff') as f:
    data = arff.load(f)
data = pd.DataFrame(data["data"], columns=[a[0] for a in data['attributes']])
data = data.drop(columns=["MOLECULE_CHEMBL_ID"])

# Data has an original order-based distribution shift, vanishes after random sampling.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=False,
    target_feature="MEDIAN_PXC50",
    skip_feature_importance_shift_detection=True,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("QSAR-TID-11.csv", index=False)
