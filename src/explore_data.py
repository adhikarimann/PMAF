import pandas as pd

# =============================================================================
# LOAD DATASET
# =============================================================================

rhs = pd.read_csv("data/raw/rhs.csv")

rhs_2021 = rhs[rhs["Year"] == "Financial Year (Apr - Mar), 2021"]

print(rhs_2021.shape)

print(rhs_2021.duplicated(subset=["State", "District"]).sum())

rhs_2021 = rhs[rhs["Year"] == "Financial Year (Apr - Mar), 2021"]

print(rhs_2021.isnull().sum())