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
    "Census": census,
    "NFHS": nfhs,
    "RHS": rhs,
    "SECC": secc,
    "RBI": rbi
}

KEYS = ["State", "District"]

# =============================================================================
# VALIDATE DATASETS
# =============================================================================

print("=" * 80)
print("DATASET VALIDATION")
print("=" * 80)

for name, df in datasets.items():

    duplicates = df.duplicated(subset=KEYS).sum()

    print(f"\n{name}")
    print("-" * 40)
    print(f"Rows                 : {len(df)}")
    print(f"Unique District Keys : {df[KEYS].drop_duplicates().shape[0]}")
    print(f"Duplicate Keys       : {duplicates}")

    if duplicates > 0:
        raise ValueError(f"Duplicate State-District keys found in {name}")

# =============================================================================
# COVERAGE COMPARISON
# =============================================================================

print("\n")
print("=" * 80)
print("DATASET COVERAGE")
print("=" * 80)

dataset_names = list(datasets.keys())

for i in range(len(dataset_names)):

    for j in range(i + 1, len(dataset_names)):

        name1 = dataset_names[i]
        name2 = dataset_names[j]

        keys1 = set(
            map(tuple,
                datasets[name1][KEYS].drop_duplicates().values)
        )

        keys2 = set(
            map(tuple,
                datasets[name2][KEYS].drop_duplicates().values)
        )

        matched = len(keys1 & keys2)
        only_first = len(keys1 - keys2)
        only_second = len(keys2 - keys1)

        print(f"\n{name1}  <-->  {name2}")
        print("-" * 40)
        print(f"Matched Districts : {matched}")
        print(f"Only in {name1:<10}: {only_first}")
        print(f"Only in {name2:<10}: {only_second}")

# =============================================================================
# MERGE DATASETS
# =============================================================================

print("\n")
print("=" * 80)
print("MERGING DATASETS")
print("=" * 80)

merged = census.copy()

merged = merged.merge(
    nfhs,
    on=KEYS,
    how="outer"
)

merged = merged.merge(
    rhs,
    on=KEYS,
    how="outer"
)

merged = merged.merge(
    secc,
    on=KEYS,
    how="outer"
)

merged = merged.merge(
    rbi,
    on=KEYS,
    how="outer"
)

# =============================================================================
# MERGED DATASET SUMMARY
# =============================================================================

print("\n")
print("=" * 80)
print("MERGED DATASET SUMMARY")
print("=" * 80)

print(f"Rows    : {merged.shape[0]}")
print(f"Columns : {merged.shape[1]}")

print("\nMissing Values")

missing = merged.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

print("\nDuplicate State-District Keys")
print(
    merged.duplicated(subset=KEYS).sum()
)

# =============================================================================
# SAVE MERGED DATASET
# =============================================================================

merged.to_csv(
    "data/merged/merged_dataset.csv",
    index=False
)

print("\n")
print("=" * 80)
print("MERGED DATASET SAVED SUCCESSFULLY")
print("=" * 80)