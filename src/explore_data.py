import pandas as pd

census = pd.read_excel("data/raw/census.xlsx")

print(census.head())
print("\n")
print(census.columns)