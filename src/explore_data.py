import pandas as pd

nfhs = pd.read_csv("data/raw/nfhs.csv")



print("\nUnique States:", nfhs["State"].nunique())
print("Unique Districts:", nfhs["District"].nunique())

print("\nUnique State-District pairs:",nfhs[["State", "District"]].drop_duplicates().shape[0])

print("\nDuplicate State-District pairs:")
print(nfhs.duplicated(subset=["State", "District"]).sum())