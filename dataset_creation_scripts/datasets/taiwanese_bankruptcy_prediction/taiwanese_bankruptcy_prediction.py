"""Download and unzip from here https://doi.org/10.24432/C5004D to `uci_data/` folder."""

from __future__ import annotations

import pandas as pd
from dataset_creation_scripts.dataset_check_utils import run_all_checks

data = pd.read_csv("uci_data/data.csv")
target_feature = "Bankrupt"
data.columns = features = [
    "Bankrupt",
    "ROA_C_Before_Interest_Depreciation",
    "ROA_A_Before_Interest_After_Tax",
    "ROA_B_Before_Interest_Depreciation_After_Tax",
    "Operating_Gross_Margin",
    "Realized_Sales_Gross_Margin",
    "Operating_Profit_Rate",
    "PreTax_Net_Interest_Rate",
    "AfterTax_Net_Interest_Rate",
    "NonIndustry_Income_Expenditure_Revenue",
    "Continuous_Interest_Rate_After_Tax",
    "Operating_Expense_Rate",
    "R&D_Expense_Rate",
    "Cash_Flow_Rate",
    "InterestBearing_Debt_Interest_Rate",
    "Tax_Rate_A",
    "Net_Value_Per_Share_B",
    "Net_Value_Per_Share_A",
    "Net_Value_Per_Share_C",
    "Persistent_EPS_Last_4_Seasons",
    "Cash_Flow_Per_Share",
    "Revenue_Per_Share",
    "Operating_Profit_Per_Share",
    "Net_Profit_Before_Tax_Per_Share",
    "Realized_Sales_Gross_Profit_Growth_Rate",
    "Operating_Profit_Growth_Rate",
    "AfterTax_Net_Profit_Growth_Rate",
    "Regular_Net_Profit_Growth_Rate",
    "Continuous_Net_Profit_Growth_Rate",
    "Total_Asset_Growth_Rate",
    "Net_Value_Growth_Rate",
    "Total_Asset_Return_Growth_Rate",
    "Cash_Reinvestment_Percent",
    "Current_Ratio",
    "Quick_Ratio",
    "Interest_Expense_Ratio",
    "Total_Debt_to_Net_Worth",
    "Debt_Ratio_Percent",
    "Net_Worth_to_Assets",
    "LongTerm_Fund_Suitability_Ratio_A",
    "Borrowing_Dependency",
    "Contingent_Liabilities_to_Net_Worth",
    "Operating_Profit_to_PaidIn_Capital",
    "Net_Profit_Before_Tax_to_PaidIn_Capital",
    "Inventory_Accounts_Receivable_to_Net_Value",
    "Total_Asset_Turnover",
    "Accounts_Receivable_Turnover",
    "Average_Collection_Days",
    "Inventory_Turnover_Rate",
    "Fixed_Assets_Turnover_Frequency",
    "Net_Worth_Turnover_Rate",
    "Revenue_Per_Person",
    "Operating_Profit_Per_Person",
    "Allocation_Rate_Per_Person",
    "Working_Capital_to_Total_Assets",
    "Quick_Assets_to_Total_Assets",
    "Current_Assets_to_Total_Assets",
    "Cash_to_Total_Assets",
    "Quick_Assets_to_Current_Liability",
    "Cash_to_Current_Liability",
    "Current_Liability_to_Assets",
    "Operating_Funds_to_Liability",
    "Inventory_to_Working_Capital",
    "Inventory_to_Current_Liability",
    "Current_Liabilities_to_Liability",
    "Working_Capital_to_Equity",
    "Current_Liabilities_to_Equity",
    "LongTerm_Liability_to_Current_Assets",
    "Retained_Earnings_to_Total_Assets",
    "Total_Income_to_Total_Expense",
    "Total_Expense_to_Assets",
    "Current_Asset_Turnover_Rate",
    "Quick_Asset_Turnover_Rate",
    "Working_Capital_Turnover_Rate",
    "Cash_Turnover_Rate",
    "Cash_Flow_to_Sales",
    "Fixed_Assets_to_Assets",
    "Current_Liability_to_Liability",
    "Current_Liability_to_Equity",
    "Equity_to_LongTerm_Liability",
    "Cash_Flow_to_Total_Assets",
    "Cash_Flow_to_Liability",
    "CFO_to_Assets",
    "Cash_Flow_to_Equity",
    "Current_Liability_to_Current_Assets",
    "Liability_Assets_Flag",
    "Net_Income_to_Total_Assets",
    "Total_Assets_to_GNP_Price",
    "NoCredit_Interval",
    "Gross_Profit_to_Sales",
    "Net_Income_to_Stockholders_Equity",
    "Liability_to_Equity",
    "DFL",
    "Interest_Coverage_Ratio",
    "Net_Income_Flag",
    "Equity_to_Liability",
]
data = data.drop(columns=["Net_Income_Flag"])
data[target_feature] = data[target_feature].map({1: "Yes", 0: "No"})

# Data is ordered (likely by years), thus dist shift for original order. Shuffling the data removes this.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

run_all_checks(
    data=data,
    classification=True,
    target_feature=target_feature,
)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.to_csv("taiwanese_bankruptcy_prediction.csv", index=False)
