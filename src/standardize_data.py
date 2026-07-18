import pandas as pd

# =============================================================================
# LOAD CLEANED DATASETS
# =============================================================================

census = pd.read_csv("data/cleaned/census_clean.csv")
nfhs = pd.read_csv("data/cleaned/nfhs_clean.csv")
rhs = pd.read_csv("data/cleaned/rhs_clean.csv")
secc = pd.read_csv("data/cleaned/secc_clean.csv")
rbi = pd.read_csv("data/cleaned/rbi_clean.csv")

datasets = {
    "Census": census,
    "NFHS": nfhs,
    "RHS": rhs,
    "SECC": secc,
    "RBI": rbi
}

# =============================================================================
# STANDARDIZE TEXT
# =============================================================================

def standardize_text(text):
    if pd.isna(text):
        return text

    text = str(text).strip()
    text = " ".join(text.split())
    text = text.title()

    return text


for df in datasets.values():
    df["State"] = df["State"].apply(standardize_text)
    df["District"] = df["District"].apply(standardize_text)

# =============================================================================
# STATE NAME MAPPING
# =============================================================================

STATE_MAPPING = {

    # Delhi
    "Nct Of Delhi": "Delhi",

    # Jammu & Kashmir
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Jammu And Kashmir": "Jammu and Kashmir",

    # Andaman & Nicobar
    "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
    "Andaman And Nicobar Islands": "Andaman and Nicobar Islands",

    # Census 2011 UTs
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
    "Daman & Diu": "Daman and Diu",

    # NOTE:
    # Do NOT convert
    # "The Dadra And Nagar Haveli And Daman And Diu"
    # because Census 2011 has two separate UTs.

}

# =============================================================================
# DISTRICT NAME MAPPING
# =============================================================================

DISTRICT_MAPPING = {

    # -------------------------
    # Jammu & Kashmir
    # -------------------------
    "Badgam": "Budgam",
    "Baramula": "Baramulla",
    "Bandipore": "Bandipora",
    "Punch": "Poonch",
    "Shupiyan": "Shopian",

    # -------------------------
    # Himachal Pradesh
    # -------------------------
    "Lahul & Spiti": "Lahaul And Spiti",

    # -------------------------
    # Punjab
    # -------------------------
    "Firozpur": "Ferozepur",

    # -------------------------
    # Haryana
    # -------------------------
    "Gurgaon": "Gurugram",
    "Mewat": "Nuh",

    # -------------------------
    # Uttarakhand
    # -------------------------
    "Hardwar": "Haridwar",
    "Udham Singh Nagar": "Udam Singh Nagar",
    "Rudraprayag": "Rudra Prayag",
    "Uttarkashi": "Uttar Kashi",

    # -------------------------
    # Bihar
    # -------------------------
    "Purba Champaran": "Purbi Champaran",

    # -------------------------
    # Jharkhand
    # -------------------------
    "Saraikela-Kharsawan": "Saraikela Kharsawan",
    "Sahibganj": "Sahebganj",

    # -------------------------
    # Karnataka
    # -------------------------
    "Bagalkot": "Bagalkote",
    "Chamarajanagar": "Chamarajanagara",
    "Davanagere": "Davangere",
    "Chikmagalur": "Chikkamagaluru",

    # -------------------------
    # Maharashtra
    # -------------------------
    "Ahmadnagar": "Ahmednagar",
    "Buldana": "Buldhana",
    "Gondiya": "Gondia",

    # -------------------------
    # Meghalaya
    # -------------------------
    "Ribhoi": "Ri Bhoi",

    # -------------------------
    # Odisha
    # -------------------------
    "Nabarangapur": "Nabarangpur",

    # -------------------------
    # Rajasthan
    # -------------------------
    "Jhunjhunun": "Jhunjhunu",
    "Jalor": "Jalore",

    # -------------------------
    # Tamil Nadu
    # -------------------------
    "Viluppuram": "Villupuram",

    # -------------------------
    # Uttar Pradesh
    # -------------------------
    "Bara Banki": "Barabanki",
    "Kushinagar": "Kushi Nagar",
    "Siddharthnagar": "Siddharth Nagar",

    # -------------------------
    # West Bengal
    # -------------------------
    "Puruliya": "Purulia",
    "Maldah": "Malda",

    # -------------------------
    # Chhattisgarh
    # -------------------------
    "Janjgir - Champa": "Janjgir-Champa",

    # -------------------------
    # Andaman & Nicobar
    # -------------------------
    "North & Middle Andaman": "North And Middle Andaman",
    "South Andaman": "South Andamans"

}

# =============================================================================
# APPLY MAPPINGS
# =============================================================================

for df in datasets.values():

    df["State"] = df["State"].replace(STATE_MAPPING)
    df["District"] = df["District"].replace(DISTRICT_MAPPING)

# =============================================================================
# VALIDATION
# =============================================================================

print("=" * 80)
print("STANDARDIZATION SUMMARY")
print("=" * 80)

for name, df in datasets.items():

    print(f"\n{name}")
    print("-" * 40)

    print(f"Rows      : {len(df)}")
    print(f"States    : {df['State'].nunique()}")
    print(f"Districts : {df['District'].nunique()}")

    print(
        "Duplicate State-District Pairs :",
        df.duplicated(subset=["State", "District"]).sum()
    )

# =============================================================================
# STATE COMPARISON
# =============================================================================

print("\n")
print("=" * 80)
print("STATE COMPARISON")
print("=" * 80)

all_states = set()

for df in datasets.values():
    all_states.update(df["State"].dropna().unique())

for state in sorted(all_states):

    presence = []

    for name, df in datasets.items():

        if state in df["State"].values:
            presence.append(name)

    print(f"{state:<45} -> {', '.join(presence)}")

# =============================================================================
# SAVE STANDARDIZED DATASETS
# =============================================================================

census.to_csv(
    "data/standardized/census_standardized.csv",
    index=False
)

nfhs.to_csv(
    "data/standardized/nfhs_standardized.csv",
    index=False
)

rhs.to_csv(
    "data/standardized/rhs_standardized.csv",
    index=False
)

secc.to_csv(
    "data/standardized/secc_standardized.csv",
    index=False
)

rbi.to_csv(
    "data/standardized/rbi_standardized.csv",
    index=False
)

print("\n")
print("=" * 80)
print("STANDARDIZED DATASETS SAVED SUCCESSFULLY")
print("=" * 80)