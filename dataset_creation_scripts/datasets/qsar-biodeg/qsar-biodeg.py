"""Download and unzip from here to https://doi.org/10.24432/C5H60M `uci_data`."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/biodeg.csv", sep=";")
target_feature = "Biodegradable"
data.columns  = [
    "Laplace_Leading_Eigenvalue",
    "Weighted_Balaban_Index_Barysz_Matrix",
    "Num_Heavy_Atoms",
    "Freq_NN_At_Dist1",
    "Freq_CN_At_Dist4",
    "Num_ssssC_Atoms",
    "Num_Substituted_BenzeneC",
    "Percentage_C_Atoms",
    "Num_Terminal_PrimaryC",
    "Num_Oxygen_Atoms",
    "Freq_CN_At_Dist3",
    "Sum_dssC_EStates",
    "Weighted_HyperWiener_Index_Burden_Matrix",
    "Lopping_Centric_Index",
    "Laplace_Spectral_Moment6",
    "Freq_CO_At_Dist3",
    "Mean_Sanderson_Electronegativity",
    "Mean_Ionization_Potential",
    "Num_N_Hydrazine",
    "Num_Aromatic_Nitro_Groups",
    "Num_CRX3",
    "Weighted_Normalized_SpectralPositiveSum_Burden_Matrix",
    "Num_Circuits",
    "Presence_CBr_At_Dist1",
    "Presence_CCl_At_Dist3",
    "N073_chemical_substructure", # no idea what this might be
    "Adjacency_LeadingEigenvalue",
    "Intrinsic_State_Pseudoconnectivity",
    "Presence_CBr_At_Dist4",
    "Sum_dO_EStates",
    "Laplace_MoharIndex2",
    "Num_RingTertiaryC",
    "C026_chemical_substructure",
    "Freq_CN_At_Dist2",
    "Num_HBond_Donors_Atoms",
    "Weighted_LeadingEigenvalue_Burden_Matrix",
    "Intrinsic_State_Pseudoconnectivity_SAvg",
    "Num_Nitrogen_Atoms",
    "Weighted_SpectralMoment6_Burden_Matrix",
    "Num_Esters",
    "Num_Halogen_Atoms",
    target_feature
]
data[target_feature] = data[target_feature].map({"RB": "Yes", "NRB": "No"})

# Shows a distribution shift, that means the original data was sorted
# by an order (by feature "Laplace_LeadingEigenvalue" and target). Vanishes if we
# shuffle here.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("qsar-biodeg.csv", index=False)
