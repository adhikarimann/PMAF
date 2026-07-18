import pandas as pd

# =============================================================================
# CLEAN RHS DATASET
# =============================================================================

# Read raw dataset
rhs = pd.read_csv("data/raw/rhs.csv")

# =============================================================================
# Filter latest financial year
# =============================================================================

rhs = rhs[rhs["Year"] == "Financial Year (Apr - Mar), 2021"].copy()

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
    # Healthcare Infrastructure
    # -------------------------------------------------------------------------
    "Functional Sub Centres (UOM:Number), Scaling Factor:1",

    "Functional Primary Health Centres (Phcs) (UOM:Number), Scaling Factor:1",

    "Functional Community Health Centres (Chcs) (UOM:Number), Scaling Factor:1",

    "Functional Sub Divisional Hospitals (Sdhs) (UOM:Number), Scaling Factor:1",

    "Functional District Hospitals (Dhs) (UOM:Number), Scaling Factor:1",
]

# Keep only required columns
rhs = rhs[required_columns].copy()

# =============================================================================
# Rename columns
# =============================================================================

rename_columns = {

    "State": "State",

    "District": "District",

    "Functional Sub Centres (UOM:Number), Scaling Factor:1":
        "Functional_Sub_Centres",

    "Functional Primary Health Centres (Phcs) (UOM:Number), Scaling Factor:1":
        "Functional_PHCs",

    "Functional Community Health Centres (Chcs) (UOM:Number), Scaling Factor:1":
        "Functional_CHCs",

    "Functional Sub Divisional Hospitals (Sdhs) (UOM:Number), Scaling Factor:1":
        "Functional_SDHs",

    "Functional District Hospitals (Dhs) (UOM:Number), Scaling Factor:1":
        "Functional_DHs",
}

rhs.rename(columns=rename_columns, inplace=True)

# =============================================================================
# Validation
# =============================================================================

print("=" * 70)
print("CLEANED RHS DATASET")
print("=" * 70)

print(f"Rows    : {rhs.shape[0]}")
print(f"Columns : {rhs.shape[1]}")

missing = rhs.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("\nNo missing values found.")
else:
    print("\nMissing Values:")
    print(missing)

print("\nDuplicate State-District Pairs:")
print(rhs.duplicated(subset=["State", "District"]).sum())

# =============================================================================
# Save cleaned dataset
# =============================================================================

rhs.to_csv("data/cleaned/rhs_clean.csv", index=False)

print("\n✅ Cleaned dataset saved to data/cleaned/rhs_clean.csv")