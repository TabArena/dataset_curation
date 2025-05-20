"""Download and unzip from here https://doi.org/10.24432/C5FS64 to `uci_data/` folder."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/data.csv")
target_feature = "Malware"
data = data.rename(
    columns={
        "Result": target_feature,
        # Rename to make short enough for OpenML
        "com.samsung.android.providers.context.permission.WRITE_USE_APP_FEATURE_SURVEY": "samsung.providers.context.permission.WRITE_USE_APP_FEATURE_SURVEY",
        "com.google.android.finsky.permission.BIND_GET_INSTALL_REFERRER_SERVICE": "google.finsky.permission.BIND_GET_INSTALL_REFERRER_SERVICE",
        "com.samsung.android.providers.context.permission.WRITE_USE_APP_FEATURE_SURVEY": "samsung.providers.context.WRITE_USE_APP_FEATURE_SURVEY",
    }
)
data[target_feature] = data[target_feature].map({0: "No", 1: "Yes"})

# Drop duplicates
data = data.drop_duplicates(keep="first")

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    n_shift_repeats=1,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("NATICUSdroid.csv", index=False)
