import pandas as pd

# =============================================================================
# CLEAN NFHS DATASET
# =============================================================================

# Read raw dataset
nfhs = pd.read_csv("data/raw/nfhs.csv")

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
    # Chronic Disease Burden
    # -------------------------------------------------------------------------
    "Women Age Group 15 To 49 Years Who Are Anaemic (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Pregnant Women Age Group 15 To 49 Years Who Are Anaemic (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Children Age Group 6 To 59 Months Who Are Anaemic (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Men Suffering From High Blood Sugar Level (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Women Suffering From High Blood Sugar Level (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Elevated Blood Pressure Or Taking Medicine To Control Blood Pressure (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Women Suffering From Elevated Blood Pressure (%) (UOM:%(Percentage)), Scaling Factor:1",

    # -------------------------------------------------------------------------
    # Acute Disease Burden
    # -------------------------------------------------------------------------
    "Prevalence Of Diarrhoea In The 2 Weeks Preceding The Survey (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Prevalence Of Symptoms Of Acute Respiratory Infection (Ari) In The 2 Weeks Preceding The Survey (%) (UOM:%(Percentage)), Scaling Factor:1",

    # -------------------------------------------------------------------------
    # Healthcare Access
    # -------------------------------------------------------------------------
    "Births Attended By Skilled Health Personnel (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Institutional Births (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Population And Household Profile-Households With Any Usual Member Covered Under A Health Insurance Or Financing Scheme (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Average Out Of Pocket Expenditure For Each Delivery In Public Health Facility (UOM:INR(IndianRupees)), Scaling Factor:1",

    # -------------------------------------------------------------------------
    # Infrastructure
    # -------------------------------------------------------------------------
    "Households Using Clean Fuel For Cooking (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Population Living In Households That Use An Improved Sanitation Facility (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Population Living In Households With An Improved Drinking Water Source (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Population Living In Households With Electricity (%) (UOM:%(Percentage)), Scaling Factor:1",

    # -------------------------------------------------------------------------
    # Education
    # -------------------------------------------------------------------------
    "Women With 10 Or More Years Of Schooling (%) (UOM:%(Percentage)), Scaling Factor:1",

    # -------------------------------------------------------------------------
    # Lifestyle
    # -------------------------------------------------------------------------
    "Men Age Group 15 Years And Above Who Consume Alcohol (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Women Age 15 Years And Above Who Consume Alcohol (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Men Age Group 15 Years And Above Who Use Any Kind Of Tobacco (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Women Age 15 Years And Above Who Use Any Kind Of Tobacco (%) (UOM:%(Percentage)), Scaling Factor:1",

    # -------------------------------------------------------------------------
    # Nutrition
    # -------------------------------------------------------------------------
    "Children Under 5 Years Who Are Underweight (Weight-For-Age) (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Children Under 5 Years Who Are Stunted (Height-For-Age) (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Children Under 5 Years Who Are Wasted (Weight-For-Height) (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Women With Body Mass Index (Bmi) Below Normal (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Women Who Are Overweight Or Obese (%)  (UOM:%(Percentage)), Scaling Factor:1",

    # -------------------------------------------------------------------------
    # Maternal Healthcare
    # -------------------------------------------------------------------------
    "Mothers Who Had An Antenatal Check-Up In The First Trimester (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Mothers Who Had At Least 4 Antenatal Care Visits (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Mothers Who Consumed Iron Folic Acid For 100 Days Or More When They Were Pregnant (%) (UOM:%(Percentage)), Scaling Factor:1",

    # -------------------------------------------------------------------------
    # Oncology Awareness
    # -------------------------------------------------------------------------
    "Women Who Have Ever Undergone A Breast Examination For Breast Cancer (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Women-Ever Undergone A Screening Test For Cervical Cancer (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Women-Ever Undergone An Oral Cavity Examination For Oral Cancer (%) (UOM:%(Percentage)), Scaling Factor:1",
]

# Keep only required columns
nfhs = nfhs[required_columns].copy()

# =============================================================================
# Rename columns
# =============================================================================

rename_columns = {

    "State": "State",
    "District": "District",

    "Women Age Group 15 To 49 Years Who Are Anaemic (%) (UOM:%(Percentage)), Scaling Factor:1": "Women_Anaemia",
    "Pregnant Women Age Group 15 To 49 Years Who Are Anaemic (%) (UOM:%(Percentage)), Scaling Factor:1": "Pregnant_Anaemia",
    "Children Age Group 6 To 59 Months Who Are Anaemic (%) (UOM:%(Percentage)), Scaling Factor:1": "Children_Anaemia",

    "Men Suffering From High Blood Sugar Level (%) (UOM:%(Percentage)), Scaling Factor:1": "Men_High_BloodSugar",
    "Women Suffering From High Blood Sugar Level (%) (UOM:%(Percentage)), Scaling Factor:1": "Women_High_BloodSugar",

    "Elevated Blood Pressure Or Taking Medicine To Control Blood Pressure (%) (UOM:%(Percentage)), Scaling Factor:1": "Men_Elevated_BP",
    "Women Suffering From Elevated Blood Pressure (%) (UOM:%(Percentage)), Scaling Factor:1": "Women_Elevated_BP",

    "Prevalence Of Diarrhoea In The 2 Weeks Preceding The Survey (%) (UOM:%(Percentage)), Scaling Factor:1": "Diarrhoea_Prevalence",
    "Prevalence Of Symptoms Of Acute Respiratory Infection (Ari) In The 2 Weeks Preceding The Survey (%) (UOM:%(Percentage)), Scaling Factor:1": "ARI_Prevalence",

    "Births Attended By Skilled Health Personnel (%) (UOM:%(Percentage)), Scaling Factor:1": "Skilled_Births",
    "Institutional Births (%) (UOM:%(Percentage)), Scaling Factor:1": "Institutional_Births",

    "Population And Household Profile-Households With Any Usual Member Covered Under A Health Insurance Or Financing Scheme (%) (UOM:%(Percentage)), Scaling Factor:1": "Health_Insurance",
    "Average Out Of Pocket Expenditure For Each Delivery In Public Health Facility (UOM:INR(IndianRupees)), Scaling Factor:1": "Out_Of_Pocket_Expense",

    "Households Using Clean Fuel For Cooking (%) (UOM:%(Percentage)), Scaling Factor:1": "Clean_Fuel",
    "Population Living In Households That Use An Improved Sanitation Facility (%) (UOM:%(Percentage)), Scaling Factor:1": "Improved_Sanitation",
    "Population Living In Households With An Improved Drinking Water Source (%) (UOM:%(Percentage)), Scaling Factor:1": "Improved_Drinking_Water",
    "Population Living In Households With Electricity (%) (UOM:%(Percentage)), Scaling Factor:1": "Electricity",

    "Women With 10 Or More Years Of Schooling (%) (UOM:%(Percentage)), Scaling Factor:1": "Women_10Plus_Schooling",

    "Men Age Group 15 Years And Above Who Consume Alcohol (%) (UOM:%(Percentage)), Scaling Factor:1": "Men_Alcohol",
    "Women Age 15 Years And Above Who Consume Alcohol (%) (UOM:%(Percentage)), Scaling Factor:1": "Women_Alcohol",

    "Men Age Group 15 Years And Above Who Use Any Kind Of Tobacco (%) (UOM:%(Percentage)), Scaling Factor:1": "Men_Tobacco",
    "Women Age 15 Years And Above Who Use Any Kind Of Tobacco (%) (UOM:%(Percentage)), Scaling Factor:1": "Women_Tobacco",

    "Children Under 5 Years Who Are Underweight (Weight-For-Age) (%) (UOM:%(Percentage)), Scaling Factor:1": "Children_Underweight",
    "Children Under 5 Years Who Are Stunted (Height-For-Age) (%) (UOM:%(Percentage)), Scaling Factor:1": "Children_Stunted",
    "Children Under 5 Years Who Are Wasted (Weight-For-Height) (%) (UOM:%(Percentage)), Scaling Factor:1": "Children_Wasted",

    "Women With Body Mass Index (Bmi) Below Normal (%) (UOM:%(Percentage)), Scaling Factor:1": "Women_Low_BMI",
    "Women Who Are Overweight Or Obese (%)  (UOM:%(Percentage)), Scaling Factor:1": "Women_Overweight",

    "Mothers Who Had An Antenatal Check-Up In The First Trimester (%) (UOM:%(Percentage)), Scaling Factor:1": "ANC_First_Trimester",
    "Mothers Who Had At Least 4 Antenatal Care Visits (%) (UOM:%(Percentage)), Scaling Factor:1": "ANC_4Plus_Visits",
    "Mothers Who Consumed Iron Folic Acid For 100 Days Or More When They Were Pregnant (%) (UOM:%(Percentage)), Scaling Factor:1": "IFA_100_Days",

    "Women Who Have Ever Undergone A Breast Examination For Breast Cancer (%) (UOM:%(Percentage)), Scaling Factor:1": "Breast_Cancer_Screening",
    "Women-Ever Undergone A Screening Test For Cervical Cancer (%) (UOM:%(Percentage)), Scaling Factor:1": "Cervical_Cancer_Screening",
    "Women-Ever Undergone An Oral Cavity Examination For Oral Cancer (%) (UOM:%(Percentage)), Scaling Factor:1": "Oral_Cancer_Screening",
}

nfhs.rename(columns=rename_columns, inplace=True)

# =============================================================================
# Validation
# =============================================================================

print("=" * 70)
print("CLEANED NFHS DATASET")
print("=" * 70)

print(f"Rows    : {nfhs.shape[0]}")
print(f"Columns : {nfhs.shape[1]}")


missing = nfhs.isnull().sum()

missing = missing[missing > 0]

if missing.empty:
    print("\nNo missing values found.")
else:
    print("\nMissing Values:")
    print(missing)

print("\nDuplicate State-District Pairs:")
print(nfhs.duplicated(subset=["State", "District"]).sum())

# =============================================================================
# Save cleaned dataset
# =============================================================================

nfhs.to_csv("data/cleaned/nfhs_clean.csv", index=False)

print("\n✅ Cleaned dataset saved to data/cleaned/nfhs_clean.csv")