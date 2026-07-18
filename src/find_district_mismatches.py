import os
import pandas as pd

# =============================================================================
# LOAD STANDARDIZED DATASETS
# =============================================================================

census = pd.read_csv("data/standardized/census_standardized.csv")
nfhs = pd.read_csv("data/standardized/nfhs_standardized.csv")
rhs = pd.read_csv("data/standardized/rhs_standardized.csv")
secc = pd.read_csv("data/standardized/secc_standardized.csv")
rbi = pd.read_csv("data/standardized/rbi_standardized.csv")

datasets = {
    "NFHS": nfhs,
    "RHS": rhs,
    "SECC": secc,
    "RBI": rbi
}

KEYS = ["State", "District"]

# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

os.makedirs("data/mismatches", exist_ok=True)

# =============================================================================
# COMPARE CENSUS WITH OTHER DATASETS
# =============================================================================

for name, df in datasets.items():

    print("\n" + "=" * 80)
    print(f"CENSUS vs {name}")
    print("=" * 80)

    census_keys = census[KEYS].drop_duplicates()
    other_keys = df[KEYS].drop_duplicates()

    # -------------------------------------------------------------------------
    # Districts only in Census
    # -------------------------------------------------------------------------

    only_census = census_keys.merge(
        other_keys,
        on=KEYS,
        how="left",
        indicator=True
    )

    only_census = only_census[
        only_census["_merge"] == "left_only"
    ].drop(columns="_merge")

    # -------------------------------------------------------------------------
    # Districts only in Other Dataset
    # -------------------------------------------------------------------------

    only_other = other_keys.merge(
        census_keys,
        on=KEYS,
        how="left",
        indicator=True
    )

    only_other = only_other[
        only_other["_merge"] == "left_only"
    ].drop(columns="_merge")

    # -------------------------------------------------------------------------
    # Display Summary
    # -------------------------------------------------------------------------

    print(f"\nOnly in Census : {len(only_census)}")
    print(f"Only in {name:<7}: {len(only_other)}")

    print("\nSample (Only in Census)")
    print("-" * 40)
    print(only_census.head(20))

    print("\nSample (Only in " + name + ")")
    print("-" * 40)
    print(only_other.head(20))

    # -------------------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------------------

    only_census.to_csv(
        f"data/mismatches/census_only_vs_{name.lower()}.csv",
        index=False
    )

    only_other.to_csv(
        f"data/mismatches/{name.lower()}_only_vs_census.csv",
        index=False
    )

print("\n" + "=" * 80)
print("ALL MISMATCH FILES SAVED SUCCESSFULLY")
print("=" * 80)