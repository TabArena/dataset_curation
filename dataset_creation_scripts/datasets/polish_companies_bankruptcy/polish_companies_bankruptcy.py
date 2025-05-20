"""Download and unzip from here https://doi.org/10.24432/C5F600 to `uci_data/` folder."""

from __future__ import annotations

import arff
import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

with open("uci_data/5year.arff") as f:
    data = arff.load(f)
data = pd.DataFrame(data["data"], columns=[x[0] for x in data["attributes"]])
target_feature = "company_bankrupt"
data.columns = feature_names = [
    "net_profit_to_total_assets",
    "total_liabilities_to_total_assets",
    "working_capital_to_total_assets",
    "current_assets_to_short_term_liabilities",
    "liquidity_days_ratio",
    "retained_earnings_to_total_assets",
    "ebit_to_total_assets",
    "book_value_equity_to_total_liabilities",
    "sales_to_total_assets",
    "equity_to_total_assets",
    "extended_profit_to_total_assets",
    "gross_profit_to_short_term_liabilities",
    "gross_profit_plus_depreciation_to_sales",
    "gross_profit_plus_interest_to_total_assets",
    "liabilities_days_ratio",
    "gross_profit_plus_depreciation_to_total_liabilities",
    "total_assets_to_total_liabilities",
    "gross_profit_to_total_assets",
    "gross_profit_to_sales",
    "inventory_days_ratio",
    "sales_growth_ratio",
    "operating_profit_to_total_assets",
    "net_profit_to_sales",
    "three_year_gross_profit_to_total_assets",
    "equity_minus_share_capital_to_total_assets",
    "net_profit_plus_depreciation_to_total_liabilities",
    "operating_profit_to_financial_expenses",
    "working_capital_to_fixed_assets",
    "log_total_assets",
    "net_liabilities_to_sales",
    "gross_profit_plus_interest_to_sales",
    "current_liabilities_days_ratio",
    "operating_expenses_to_short_term_liabilities",
    "operating_expenses_to_total_liabilities",
    "sales_profit_to_total_assets",
    "total_sales_to_total_assets",
    "current_assets_minus_inventories_to_long_term_liabilities",
    "constant_capital_to_total_assets",
    "sales_profit_to_sales",
    "liquid_assets_to_short_term_liabilities",
    "liabilities_to_adjusted_operating_profit",
    "operating_profit_to_sales",
    "receivables_plus_inventory_turnover_days",
    "receivables_days_ratio",
    "net_profit_to_inventory",
    "current_assets_minus_inventory_to_short_term_liabilities",
    "inventory_days_cost_ratio",
    "ebitda_to_total_assets",
    "ebitda_to_sales",
    "current_assets_to_total_liabilities",
    "short_term_liabilities_to_total_assets",
    "short_term_liabilities_days_cost_ratio",
    "equity_to_fixed_assets",
    "constant_capital_to_fixed_assets",
    "working_capital_absolute",
    "gross_margin",
    "adjusted_liquidity_ratio",
    "total_costs_to_total_sales",
    "long_term_liabilities_to_equity",
    "inventory_turnover_ratio",
    "receivables_turnover_ratio",
    "short_term_liabilities_days_ratio",
    "sales_to_short_term_liabilities",
    "sales_to_fixed_assets",
    target_feature,
]
data = data.rename(columns={"target": target_feature})
data[target_feature] = data[target_feature].map({"1": "Yes", "0": "No"})

# Data is ordered, thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
    n_shift_repeats=1,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("polish_companies_bankruptcy.csv", index=False)
