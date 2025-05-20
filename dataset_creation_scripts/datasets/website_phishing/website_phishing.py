"""Download and unzip from here https://doi.org/10.24432/C5B301 to `uci_data/` folder."""

from __future__ import annotations

import arff
import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

with open("uci_data/PhishingData.arff") as f:
    data = arff.load(f)

data = pd.DataFrame(data["data"], columns=[x[0] for x in data["attributes"]])
# according to description this is the encoding for all the features and the target
encoding_map = {"1": "Legitimate", "0": "Suspicious", "-1": "Phishy"}
data = data.map(lambda x: encoding_map[x])
target_feature = "WebsiteType"
data = data.rename(columns={"Result": target_feature})

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("website_phishing.csv", index=False)
