import pandas as pd

# =============================================================================
# CLEAN RBI DATASET
# =============================================================================

# Read raw dataset
rbi = pd.read_csv("data/raw/rbi.csv")

# =============================================================================
# Filter Latest Available Data
# =============================================================================

rbi = rbi[
    (rbi["Year"] == "Financial Year (Apr - Mar), 2024") &
    (rbi["Quarter"] == "January - March, 2025")
].copy()

# =============================================================================
# Columns to Retain
# =============================================================================

required_columns = [

    # -------------------------------------------------------------------------
    # Identifiers
    # -------------------------------------------------------------------------
    "State",
    "District",

    # -------------------------------------------------------------------------
    # Banking Indicators
    # -------------------------------------------------------------------------
    "Reporting Offices (UOM:Number), Scaling Factor:1",

    "Aggregate Deposit (UOM:INR(IndianRupees)), Scaling Factor:10000000",

    "Gross Bank Credit (UOM:INR(IndianRupees)), Scaling Factor:10000000",
]

rbi = rbi[required_columns].copy()

# =============================================================================
# Rename Columns
# =============================================================================

rename_columns = {

    "State": "State",

    "District": "District",

    "Reporting Offices (UOM:Number), Scaling Factor:1":
        "Reporting_Offices",

    "Aggregate Deposit (UOM:INR(IndianRupees)), Scaling Factor:10000000":
        "Aggregate_Deposit",

    "Gross Bank Credit (UOM:INR(IndianRupees)), Scaling Factor:10000000":
        "Gross_Bank_Credit",
}

rbi.rename(columns=rename_columns, inplace=True)

# =============================================================================
# Validation
# =============================================================================

print("=" * 70)
print("CLEANED RBI DATASET")
print("=" * 70)

print(f"Rows    : {rbi.shape[0]}")
print(f"Columns : {rbi.shape[1]}")

missing = rbi.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("\nNo missing values found.")
else:
    print("\nMissing Values:")
    print(missing)

print("\nDuplicate State-District Pairs:")
print(rbi.duplicated(subset=["State", "District"]).sum())

# =============================================================================
# Save Cleaned Dataset
# =============================================================================

rbi.to_csv("data/cleaned/rbi_clean.csv", index=False)

print("\n✅ Cleaned dataset saved to data/cleaned/rbi_clean.csv")