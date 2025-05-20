"""The script assumes that you downloaded the data from the UCI repo and
unzipped it into directory called `annealing` in the current working directory.
"""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("./annealing/anneal.data", header=None)
test_data = pd.read_csv("./annealing/anneal.test", header=None)

# Concatenate the two datasets
combined_data = pd.concat([data, test_data], ignore_index=True)
feature_names = [
    "family",
    "product-type",
    "steel",
    "carbon",
    "hardness",
    "temper_rolling",
    "condition",
    "formability",
    "strength",
    "non-ageing",
    "surface-finish",
    "surface-quality",
    "enamelability",
    "bc",
    "bf",
    "bt",
    "bw_me",  # original: "bw/me"
    "bl",
    "m",
    "chrom",
    "phos",
    "cbond",
    "marvi",
    "exptl",
    "ferro",
    "corr",
    "blue_bright_varn_clean",  # original: "blue/bright/varnish/clean"
    "lustre",
    "jurofm",
    "s",
    "p",
    "shape",
    "thick",
    "width",
    "len",
    "oil",
    "bore",
    "packing",
    "classes",
]
combined_data.columns = feature_names

combined_data = combined_data.replace("?", "not_applicable")

# -- Dataset checks
run_all_checks(
    data=combined_data,
    classification=True,
    target_feature="classes",
)

combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
combined_data.to_csv("anneal.csv", index=False)
