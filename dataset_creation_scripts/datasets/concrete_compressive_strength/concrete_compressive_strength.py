"""Download and unzip from here https://doi.org/10.24432/C5PK67 to `uci_data/` folder."""


from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_excel("uci_data/Concrete_Data.xls")
target_feature = "ConcreteCompressiveStrength"
data.columns = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAggregate",
    "FineAggregate",
    "Age",
    target_feature
]

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("concrete_compressive_strength.csv", index=False)

