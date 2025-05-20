"""Download and unzip from here https://doi.org/10.24432/C5QW3H to `uci_data/` folder."""


from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/CASP.csv")
target_feature = "ResidualSize"
data.columns = [
    target_feature,
    "TotalSurfaceArea",
    "NonPolarExposedArea",
    "FracExposedNonPolarResidue",
    "FracExposedNonPolarPart",
    "MassWeightedExposedArea",
    "AvgDeviationExposedArea",
    "EuclideanDistance",
    "SecondaryStructurePenalty",
    "SpatialDistNK",
]

run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("physiochemical_protein.csv", index=False)

