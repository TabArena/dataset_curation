"""Download and unzip from here to https://doi.org/10.24432/C5230J `uci_data`."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/diabetic_data.csv")

data = data.drop_duplicates(subset="patient_nbr")
data = data.drop(columns=["encounter_id", "patient_nbr"])
inverse_mappings = {
    "admission_type_id": {
        1: "Emergency",
        2: "Urgent",
        3: "Elective",
        4: "Newborn",
        5: "Not Available",
        6: "NULL",
        7: "Trauma Center",
        8: "Not Mapped",
    },
    "discharge_disposition_id": {
        1: "Discharged to home",
        2: "Discharged/transferred to another short term hospital",
        3: "Discharged/transferred to SNF",
        4: "Discharged/transferred to ICF",
        5: "Discharged/transferred to another type of inpatient care institution",
        6: "Discharged/transferred to home with home health service",
        7: "Left AMA",
        8: "Discharged/transferred to home under care of Home IV provider",
        9: "Admitted as an inpatient to this hospital",
        10: "Neonate discharged to another hospital for neonatal aftercare",
        11: "Expired",
        12: "Still patient or expected to return for outpatient services",
        13: "Hospice / home",
        14: "Hospice / medical facility",
        15: "Discharged/transferred within this institution to Medicare approved swing bed",
        16: "Discharged/transferred/referred another institution for outpatient services",
        17: "Discharged/transferred/referred to this institution for outpatient services",
        18: "NULL",
        19: "Expired at home. Medicaid only, hospice.",
        20: "Expired in a medical facility. Medicaid only, hospice.",
        21: "Expired, place unknown. Medicaid only, hospice.",
        22: "Discharged/transferred to another rehab fac including rehab units of a hospital .",
        23: "Discharged/transferred to a long term care hospital.",
        24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.",
        25: "Not Mapped",
        26: "Unknown/Invalid",
        27: "Discharged/transferred to a federal health care facility.",
        28: "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
        29: "Discharged/transferred to a Critical Access Hospital (CAH).",
        30: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
    },
    "admission_source_id": {
        1: "Physician Referral",
        2: "Clinic Referral",
        3: "HMO Referral",
        4: "Transfer from a hospital",
        5: "Transfer from a Skilled Nursing Facility (SNF)",
        6: "Transfer from another health care facility",
        7: "Emergency Room",
        8: "Court/Law Enforcement",
        9: "Not Available",
        10: "Transfer from critial access hospital",
        11: "Normal Delivery",
        12: "Premature Delivery",
        13: "Sick Baby",
        14: "Extramural Birth",
        15: "Not Available",
        17: "NULL",
        18: "Transfer From Another Home Health Agency",
        19: "Readmission to Same Home Health Agency",
        20: "Not Mapped",
        21: "Unknown/Invalid",
        22: "Transfer from hospital inpt/same fac reslt in a sep claim",
        23: "Born inside this hospital",
        24: "Born outside this hospital",
        25: "Transfer from Ambulatory Surgery Center",
        26: "Transfer from Hospice",
    },
}
for feature, inverse_map in inverse_mappings.items():
    data[feature] = data[feature].map(inverse_map)

target_feature = "EarlyReadmission"
data = data.rename(columns={"readmitted": target_feature})
label_mask = data[target_feature] == "<30"
data.loc[label_mask, target_feature] = "Yes"
data.loc[~label_mask, target_feature] = "No"

# Shows a distribution shift based on the original order (which is gone after random
# shuffling)
run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    n_shift_repeats=1,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# Check after shuffle.
run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    n_shift_repeats=1,
)
data.to_csv("Diabetes130US.csv", index=False)
