import os
import pandas as pd
from difflib import SequenceMatcher

# =============================================================================
# LOAD DATASETS
# =============================================================================

census = pd.read_csv("data/standardized/census_standardized.csv")

datasets = {
    "NFHS": pd.read_csv("data/standardized/nfhs_standardized.csv"),
    "RHS": pd.read_csv("data/standardized/rhs_standardized.csv"),
    "SECC": pd.read_csv("data/standardized/secc_standardized.csv"),
    "RBI": pd.read_csv("data/standardized/rbi_standardized.csv")
}

# =============================================================================
# PARAMETERS
# =============================================================================

SIMILARITY_THRESHOLD = 0.80

os.makedirs("data/mismatches", exist_ok=True)

all_suggestions = []

# =============================================================================
# FIND CANDIDATES
# =============================================================================

for dataset_name, other in datasets.items():

    print("\n" + "=" * 80)
    print(f"CENSUS vs {dataset_name}")
    print("=" * 80)

    common_states = sorted(
        set(census["State"]).intersection(set(other["State"]))
    )

    dataset_suggestions = []

    for state in common_states:

        census_districts = sorted(
            census[census["State"] == state]["District"].unique()
        )

        other_districts = sorted(
            other[other["State"] == state]["District"].unique()
        )

        matched = set(census_districts).intersection(other_districts)

        census_only = [
            d for d in census_districts
            if d not in matched
        ]

        other_only = [
            d for d in other_districts
            if d not in matched
        ]

        for c in census_only:

            best_match = None
            best_score = 0

            for o in other_only:

                score = SequenceMatcher(
                    None,
                    c.lower(),
                    o.lower()
                ).ratio()

                if score > best_score:
                    best_score = score
                    best_match = o

            if best_score >= SIMILARITY_THRESHOLD:

                dataset_suggestions.append({

                    "Dataset": dataset_name,
                    "State": state,
                    "Census_District": c,
                    "Suggested_District": best_match,
                    "Similarity": round(best_score, 3)

                })

    suggestions = pd.DataFrame(dataset_suggestions)

    suggestions = suggestions.sort_values(
        by=["Similarity", "State"],
        ascending=[False, True]
    )

    print(f"\nSuggestions Found: {len(suggestions)}")

    if not suggestions.empty:
        print(suggestions.head(30))

    suggestions.to_csv(
        f"data/mismatches/suggestions_{dataset_name.lower()}.csv",
        index=False
    )

    all_suggestions.append(suggestions)

# =============================================================================
# COMBINE
# =============================================================================

combined = pd.concat(
    all_suggestions,
    ignore_index=True
)

combined = combined.sort_values(
    by=["Similarity", "State"],
    ascending=[False, True]
)

combined.to_csv(
    "data/mismatches/all_suggested_mappings.csv",
    index=False
)

print("\n" + "=" * 80)
print("TOTAL SUGGESTIONS:", len(combined))
print("=" * 80)

print("\nTop 50 Suggestions\n")

print(combined.head(50))

print("\n")
print("=" * 80)
print("Suggestion files saved in data/mismatches/")
print("=" * 80)