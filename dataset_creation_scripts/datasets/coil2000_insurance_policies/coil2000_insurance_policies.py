"""Download and unzip from here https://doi.org/10.24432/C5630S to `uci_data/` folder."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/ticdata2000.txt", sep="\t", header=None)
val_data = pd.concat(
    [
        pd.read_csv("uci_data/ticeval2000.txt", sep="\t", header=None),
        pd.read_csv("uci_data/tictgts2000.txt", sep="\t", header=None).rename(
            columns={0: 85}
        ),
    ],
    axis=1,
)
data = pd.concat([data, val_data], axis=0, ignore_index=True)

target_feature = "MobileHomePolicy"
data.columns = [
    "customerSubtype",
    "numberOfHouses",
    "avgSizeHousehold",
    "avgAge",
    "customerMainType",
    "romanCatholic",
    "protestant",
    "otherReligion",
    "noReligion",
    "married",
    "livingTogether",
    "otherRelation",
    "singles",
    "householdWithoutChildren",
    "householdWithChildren",
    "highLevelEducation",
    "mediumLevelEducation",
    "lowerLevelEducation",
    "highStatus",
    "entrepreneur",
    "farmer",
    "middleManagement",
    "skilledLabourers",
    "unskilledLabourers",
    "socialClassA",
    "socialClassB1",
    "socialClassB2",
    "socialClassC",
    "socialClassD",
    "rentedHouse",
    "homeOwners",
    "oneCar",
    "twoCars",
    "noCar",
    "nationalHealthService",
    "privateHealthInsurance",
    "incomeLessThan30k",
    "income30To45k",
    "income45To75k",
    "income75To122k",
    "incomeAbove123k",
    "averageIncome",
    "purchasingPowerClass",
    "contributionPrivateThirdPartyInsurance",
    "contributionThirdPartyInsuranceFirms",
    "contributionThirdPartyInsuranceAgriculture",
    "contributionCarPolicies",
    "contributionDeliveryVanPolicies",
    "contributionMotorcycleScooterPolicies",
    "contributionLorryPolicies",
    "contributionTrailerPolicies",
    "contributionTractorPolicies",
    "contributionAgriculturalMachinesPolicies",
    "contributionMopedPolicies",
    "contributionLifeInsurances",
    "contributionPrivateAccidentInsurancePolicies",
    "contributionFamilyAccidentsInsurancePolicies",
    "contributionDisabilityInsurancePolicies",
    "contributionFirePolicies",
    "contributionSurfboardPolicies",
    "contributionBoatPolicies",
    "contributionBicyclePolicies",
    "contributionPropertyInsurancePolicies",
    "contributionSocialSecurityInsurancePolicies",
    "numberOfPrivateThirdPartyInsurance",
    "numberOfThirdPartyInsuranceFirms",
    "numberOfThirdPartyInsuranceAgriculture",
    "numberOfCarPolicies",
    "numberOfDeliveryVanPolicies",
    "numberOfMotorcycleScooterPolicies",
    "numberOfLorryPolicies",
    "numberOfTrailerPolicies",
    "numberOfTractorPolicies",
    "numberOfAgriculturalMachinesPolicies",
    "numberOfMopedPolicies",
    "numberOfLifeInsurances",
    "numberOfPrivateAccidentInsurancePolicies",
    "numberOfFamilyAccidentsInsurancePolicies",
    "numberOfDisabilityInsurancePolicies",
    "numberOfFirePolicies",
    "numberOfSurfboardPolicies",
    "numberOfBoatPolicies",
    "numberOfBicyclePolicies",
    "numberOfPropertyInsurancePolicies",
    "numberOfSocialSecurityInsurancePolicies",
    target_feature,
]

data["customerSubtype"] = data["customerSubtype"].map(
    {
        1: "High Income, expensive child",
        2: "Very Important Provincials",
        3: "High status seniors",
        4: "Affluent senior apartments",
        5: "Mixed seniors",
        6: "Career and childcare",
        7: "Dinki's (double income no kids)",
        8: "Middle class families",
        9: "Modern, complete families",
        10: "Stable family",
        11: "Family starters",
        12: "Affluent young families",
        13: "Young all american family",
        14: "Junior cosmopolitan",
        15: "Senior cosmopolitans",
        16: "Students in apartments",
        17: "Fresh masters in the city",
        18: "Single youth",
        19: "Suburban youth",
        20: "Etnically diverse",
        21: "Young urban have-nots",
        22: "Mixed apartment dwellers",
        23: "Young and rising",
        24: "Young, low educated",
        25: "Young seniors in the city",
        26: "Own home elderly",
        27: "Seniors in apartments",
        28: "Residential elderly",
        29: "Porchless seniors: no front yard",
        30: "Religious elderly singles",
        31: "Low income catholics",
        32: "Mixed seniors",
        33: "Lower class large families",
        34: "Large family, employed child",
        35: "Village families",
        36: "Couples with teens 'Married with children'",
        37: "Mixed small town dwellers",
        38: "Traditional families",
        39: "Large religous families",
        40: "Large family farms",
        41: "Mixed rurals",
    }
)

data["avgAge"] = data["avgAge"].map(
    {
        1: "20-30 years",
        2: "30-40 years",
        3: "40-50 years",
        4: "50-60 years",
        5: "60-70 years",
        6: "70-80 years",
    }
)

data["customerMainType"] = data["customerMainType"].map(
    {
        1: "Successful hedonists",
        2: "Driven Growers",
        3: "Average Family",
        4: "Career Loners",
        5: "Living well",
        6: "Cruising Seniors",
        7: "Retired and Religeous",
        8: "Family with grown ups",
        9: "Conservative families",
        10: "Farmers",
    }
)

data["romanCatholic"] = data["romanCatholic"].map(
    {
        0: "0%",
        1: "1 - 10%",
        2: "11 - 23%",
        3: "24 - 36%",
        4: "37 - 49%",
        5: "50 - 62%",
        6: "63 - 75%",
        7: "76 - 88%",
        8: "89 - 99%",
        9: "100%",
    }
)

data["contributionPrivateThirdPartyInsurance"] = data[
    "contributionPrivateThirdPartyInsurance"
].map(
    {
        0: "f 0",
        1: "f 1 - 49",
        2: "f 50 - 99",
        3: "f 100 - 199",
        4: "f 200 - 499",
        5: "f 500 - 999",
        6: "f 1000 - 4999",
        7: "f 5000 - 9999",
        8: "f 10.000 - 19.999",
        9: "f 20.000 - ?",
    }
)
data[target_feature] = data[target_feature].map({0: "No", 1: "Yes"})


run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    n_shift_repeats=1,
    custom_data_split_for_shift=-4000,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("coil2000_insurance_policies.csv", index=False)
