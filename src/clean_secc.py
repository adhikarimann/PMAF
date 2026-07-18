import pandas as pd

# =============================================================================
# CLEAN SECC DATASET
# =============================================================================

# Read raw dataset
secc = pd.read_csv("data/raw/secc.csv")

# =============================================================================
# Columns to retain
# =============================================================================

required_columns = [

    # -------------------------------------------------------------------------
    # Identifiers
    # -------------------------------------------------------------------------
    "State",
    "District",

    # -------------------------------------------------------------------------
    # Market Size
    # -------------------------------------------------------------------------
    "Households (UOM:Number), Scaling Factor:1",

    # -------------------------------------------------------------------------
    # Socioeconomic Indicators
    # -------------------------------------------------------------------------
    "Landless Households Deriving Major Part Of Their Income From Manual Casual Labour (UOM:Number), Scaling Factor:1",

    "Households With Non-Agricultural Enterprises Registered With Government (UOM:Number), Scaling Factor:1",

    "Households Paying Income Tax Or Professional Tax  (UOM:Number), Scaling Factor:1",

    "Households With Salaried Job In Government (UOM:Number), Scaling Factor:1",

    "Households With Salaried Job In Private (UOM:Number), Scaling Factor:1",

    "Households With Monthly Income Of Highest Earning Member Greater Than 10000 (UOM:Number), Scaling Factor:1",

    "Households Owning Refrigerator (UOM:Number), Scaling Factor:1",

    "Households Having Four Wheeler Vehicle (UOM:Number), Scaling Factor:1",

    "Households Having Kisan Credit Card With The Credit Limit Of 50000 Rupees And Above (UOM:Number), Scaling Factor:1",
]

# Keep only required columns
secc = secc[required_columns].copy()

# =============================================================================
# Rename columns
# =============================================================================

rename_columns = {

    "State": "State",

    "District": "District",

    "Households (UOM:Number), Scaling Factor:1":
        "Households",

    "Landless Households Deriving Major Part Of Their Income From Manual Casual Labour (UOM:Number), Scaling Factor:1":
        "Landless_Manual_Labour",

    "Households With Non-Agricultural Enterprises Registered With Government (UOM:Number), Scaling Factor:1":
        "Non_Agricultural_Enterprise",

    "Households Paying Income Tax Or Professional Tax  (UOM:Number), Scaling Factor:1":
        "Income_Tax_Households",

    "Households With Salaried Job In Government (UOM:Number), Scaling Factor:1":
        "Government_Salaried",

    "Households With Salaried Job In Private (UOM:Number), Scaling Factor:1":
        "Private_Salaried",

    "Households With Monthly Income Of Highest Earning Member Greater Than 10000 (UOM:Number), Scaling Factor:1":
        "Income_Above_10000",

    "Households Owning Refrigerator (UOM:Number), Scaling Factor:1":
        "Refrigerator",

    "Households Having Four Wheeler Vehicle (UOM:Number), Scaling Factor:1":
        "Four_Wheeler",

    "Households Having Kisan Credit Card With The Credit Limit Of 50000 Rupees And Above (UOM:Number), Scaling Factor:1":
        "Kisan_Credit_Card",
}

secc.rename(columns=rename_columns, inplace=True)

# =============================================================================
# Validation
# =============================================================================

print("=" * 70)
print("CLEANED SECC DATASET")
print("=" * 70)

print(f"Rows    : {secc.shape[0]}")
print(f"Columns : {secc.shape[1]}")

missing = secc.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("\nNo missing values found.")
else:
    print("\nMissing Values:")
    print(missing)

print("\nDuplicate State-District Pairs:")
print(secc.duplicated(subset=["State", "District"]).sum())

# =============================================================================
# Save cleaned dataset
# =============================================================================

secc.to_csv("data/cleaned/secc_clean.csv", index=False)

print("\n✅ Cleaned dataset saved to data/cleaned/secc_clean.csv")