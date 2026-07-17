import pandas as pd

# ==========================================================
# PMAF Project
# Module: Clean Census Dataset
# ==========================================================

# Read raw Census dataset
census = pd.read_excel("data/raw/census.xlsx")

# ----------------------------------------------------------
# Keep only District-level TOTAL records
# ----------------------------------------------------------
districts = census[
    (census["Level"] == "DISTRICT") &
    (census["TRU"] == "Total")
].copy()

# ----------------------------------------------------------
# Select only required columns
# ----------------------------------------------------------
districts = districts[
    [
        "State",
        "District",
        "Name",
        "No_HH",
        "TOT_P",
        "P_LIT"
    ]
]

# ----------------------------------------------------------
# Rename columns
# ----------------------------------------------------------
districts.rename(
    columns={
        "Name": "District_Name",
        "No_HH": "Households",
        "TOT_P": "Population",
        "P_LIT": "Literate_Population"
    },
    inplace=True
)

# ----------------------------------------------------------
# Create Literacy Rate (%)
# ----------------------------------------------------------
districts["Literacy_Rate"] = (
    districts["Literate_Population"] /
    districts["Population"] * 100
).round(2)

# Remove intermediate column
districts.drop(columns=["Literate_Population"], inplace=True)

# Reset index
districts.reset_index(drop=True, inplace=True)

# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------
print("=" * 60)
print("CLEANED CENSUS DATASET")
print("=" * 60)

print("\nFirst 5 Rows:")
print(districts.head())

print("\nDataset Shape:")
print(districts.shape)

print("\nUnique States:")
print(districts["State"].nunique())

print("Unique District Codes:")
print(districts[["State", "District"]].drop_duplicates().shape[0])

print("\nMissing Values:")
print(districts.isnull().sum())

# ----------------------------------------------------------
# Save cleaned dataset
# ----------------------------------------------------------
districts.to_csv(
    "data/cleaned/census_clean.csv",
    index=False
)

print("\n✅ census_clean.csv saved successfully!")