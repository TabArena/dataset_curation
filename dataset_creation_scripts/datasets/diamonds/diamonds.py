"""Download https://github.com/tidyverse/ggplot2/blob/main/data-raw/diamonds.csv and name it `original_diamonds.csv`"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("original_diamonds.csv")
target_feature = "price"


# Shows a distribution shift based on the original order (which is gone after random
# shuffling) because the data is sorted by price
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
run_all_checks(
    data=data,
    classification=False,
    target_feature=target_feature,
)
data.to_csv("diamonds.csv", index=False)
