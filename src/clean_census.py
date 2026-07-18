import pandas as pd

# ==========================================================
# PMAF Project
# Module: Clean Census Dataset
# ==========================================================

# Read raw Census dataset
census = pd.read_excel("data/raw/census.xlsx")

# ==========================================================
# Create State Code -> State Name Mapping
# ==========================================================

state_mapping = (
    census[
        (census["Level"] == "STATE") &
        (census["TRU"] == "Total")
    ][["State", "Name"]]
    .drop_duplicates()
)

state_mapping.rename(
    columns={"Name": "State_Name"},
    inplace=True
)

# ==========================================================
# Keep only District-level TOTAL records
# ==========================================================

districts = census[
    (census["Level"] == "DISTRICT") &
    (census["TRU"] == "Total")
].copy()

# ==========================================================
# Merge State Names
# ==========================================================

districts = districts.merge(
    state_mapping,
    on="State",
    how="left"
)

# ==========================================================
# Select Required Columns
# ==========================================================

districts = districts[
    [
        "State_Name",
        "Name",
        "No_HH",
        "TOT_P",
        "P_LIT"
    ]
]

# ==========================================================
# Rename Columns
# ==========================================================

districts.rename(
    columns={
        "State_Name": "State",
        "Name": "District",
        "No_HH": "Households",
        "TOT_P": "Population",
        "P_LIT": "Literate_Population"
    },
    inplace=True
)

# ==========================================================
# Create Literacy Rate
# ==========================================================

districts["Literacy_Rate"] = (
    districts["Literate_Population"] /
    districts["Population"] * 100
).round(2)

districts.drop(
    columns=["Literate_Population"],
    inplace=True
)

districts.reset_index(drop=True, inplace=True)

# ==========================================================
# Validation
# ==========================================================

print("=" * 70)
print("CLEANED CENSUS DATASET")
print("=" * 70)

print(f"Rows    : {districts.shape[0]}")
print(f"Columns : {districts.shape[1]}")

print(f"\nUnique States    : {districts['State'].nunique()}")
print(f"Unique Districts : {districts['District'].nunique()}")

missing = districts.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("\nNo missing values found.")
else:
    print("\nMissing Values:")
    print(missing)

print("\nDuplicate State-District Pairs:")
print(
    districts.duplicated(
        subset=["State", "District"]
    ).sum()
)

# ==========================================================
# Save Cleaned Dataset
# ==========================================================

districts.to_csv(
    "data/cleaned/census_clean.csv",
    index=False
)

print("\n✅ census_clean.csv saved successfully!")