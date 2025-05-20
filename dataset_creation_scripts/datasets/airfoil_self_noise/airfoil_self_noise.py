"""Download and unzip from here https://doi.org/10.24432/C5VW2C to `uci_data/` folder."""


from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/airfoil_self_noise.dat", header=None, delim_whitespace=True)
target_feature = "scaled-sound-pressure"
data.columns = [
    "frequency",
    "attack-angle",
    "chord-length",
    "free-stream-velocity",
    "suction-side-displacement-thickness",
    target_feature
]

# Data is ordered by chord-length, thus dist shift for original order
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("airfoil_self_noise.csv", index=False)

