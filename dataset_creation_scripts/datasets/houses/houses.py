"""The script assumes that you downloaded the `houses.zip` from the https://lib.stat.cmu.edu/datasets/
repo and unzipped it into directory called `lib_stat_data` in the current working directory.

Furthermore, the data requires manual preprocessing:
- Remove all text from `cadata.txt` and transform it into a file containing only the data.
- Call this new file `cadata_manual.txt`.
- Remove the first two whitespaces from all rows.
- Add a header row "MedianHouseValue  MedianIncome  HousingMedianAge  TotalRooms  TotalBedrooms  Population  Households  Latitude  Longitude"
"""


from __future__ import annotations

import numpy as np
import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("lib_stat_data/cadata_manual.txt", index_col=False, sep="  ")

# Fix lat/logitude whitespace error from original data
data[["Latitude", "Longitude"]] = data["Latitude"].str.split(" ", expand=True)
data[["Latitude", "Longitude"]] = data[["Latitude", "Longitude"]].astype(float)

target_feature = "LnMedianHouseValue"

# Transform to log space as defined by original task
data[target_feature] = np.log(data["MedianHouseValue"])
data = data.drop(columns=["MedianHouseValue"])

# Spots a distribution shift in the dataset based on the order of the samples, removed by moving the shuffle here
run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("houses.csv", index=False)

