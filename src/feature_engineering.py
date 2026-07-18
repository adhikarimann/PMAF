import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import time


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    """
    Configure logging to file and console.
    
    Args:
        log_path: Path to log file.
        
    Returns:
        Configured logger instance.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("PMAF")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

CENSUS_FEATURES: List[str] = [
    "Households_x",
    "Population",
    "Literacy_Rate"
]

NFHS_FEATURES: List[str] = [
    "Women_Anaemia",
    "Pregnant_Anaemia",
    "Children_Anaemia",
    "Men_High_BloodSugar",
    "Women_High_BloodSugar",
    "Men_Elevated_BP",
    "Women_Elevated_BP",
    "Diarrhoea_Prevalence",
    "ARI_Prevalence",
    "Skilled_Births",
    "Institutional_Births",
    "Health_Insurance",
    "Out_Of_Pocket_Expense",
    "Clean_Fuel",
    "Improved_Sanitation",
    "Improved_Drinking_Water",
    "Electricity",
    "Women_10Plus_Schooling",
    "Men_Alcohol",
    "Women_Alcohol",
    "Men_Tobacco",
    "Women_Tobacco",
    "Children_Underweight",
    "Children_Stunted",
    "Children_Wasted",
    "Women_Low_BMI",
    "Women_Overweight",
    "ANC_First_Trimester",
    "ANC_4Plus_Visits",
    "IFA_100_Days",
    "Breast_Cancer_Screening",
    "Cervical_Cancer_Screening",
    "Oral_Cancer_Screening"
]

RHS_FEATURES: List[str] = [
    "Functional_Sub_Centres",
    "Functional_PHCs",
    "Functional_CHCs",
    "Functional_SDHs",
    "Functional_DHs"
]

SECC_FEATURES: List[str] = [
    "Landless_Manual_Labour",
    "Non_Agricultural_Enterprise",
    "Income_Tax_Households",
    "Government_Salaried",
    "Private_Salaried",
    "Income_Above_10000",
    "Refrigerator",
    "Four_Wheeler",
    "Kisan_Credit_Card"
]

RBI_FEATURES: List[str] = [
    "Reporting_Offices",
    "Aggregate_Deposit",
    "Gross_Bank_Credit"
]

KEY_COLUMNS: List[str] = ["State", "District"]

ALL_FEATURE_COLUMNS = (
    CENSUS_FEATURES + NFHS_FEATURES + RHS_FEATURES + SECC_FEATURES + RBI_FEATURES
)

FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "Households_x": {
        "source": "Census",
        "category": "Demographic",
        "description": "Number of households",
        "direction": "positive"
    },
    "Population": {
        "source": "Census",
        "category": "Demographic",
        "description": "Total population",
        "direction": "neutral"
    },
    "Literacy_Rate": {
        "source": "Census",
        "category": "Education",
        "description": "Percentage of literate population",
        "direction": "positive"
    },
    "Women_Anaemia": {
        "source": "NFHS",
        "category": "Health Conditions",
        "description": "Percentage of women with anaemia",
        "direction": "negative"
    },
    "Pregnant_Anaemia": {
        "source": "NFHS",
        "category": "Maternal Health",
        "description": "Percentage of pregnant women with anaemia",
        "direction": "negative"
    },
    "Children_Anaemia": {
        "source": "NFHS",
        "category": "Health Conditions",
        "description": "Percentage of children with anaemia",
        "direction": "negative"
    },
    "Men_High_BloodSugar": {
        "source": "NFHS",
        "category": "Health Conditions",
        "description": "Percentage of men with high blood sugar",
        "direction": "negative"
    },
    "Women_High_BloodSugar": {
        "source": "NFHS",
        "category": "Health Conditions",
        "description": "Percentage of women with high blood sugar",
        "direction": "negative"
    },
    "Men_Elevated_BP": {
        "source": "NFHS",
        "category": "Health Conditions",
        "description": "Percentage of men with elevated blood pressure",
        "direction": "negative"
    },
    "Women_Elevated_BP": {
        "source": "NFHS",
        "category": "Health Conditions",
        "description": "Percentage of women with elevated blood pressure",
        "direction": "negative"
    },
    "Diarrhoea_Prevalence": {
        "source": "NFHS",
        "category": "Disease Prevalence",
        "description": "Prevalence of diarrhoea",
        "direction": "negative"
    },
    "ARI_Prevalence": {
        "source": "NFHS",
        "category": "Disease Prevalence",
        "description": "Prevalence of acute respiratory infection",
        "direction": "negative"
    },
    "Skilled_Births": {
        "source": "NFHS",
        "category": "Maternal Health",
        "description": "Percentage of births attended by skilled birth attendants",
        "direction": "positive"
    },
    "Institutional_Births": {
        "source": "NFHS",
        "category": "Maternal Health",
        "description": "Percentage of institutional deliveries",
        "direction": "positive"
    },
    "Health_Insurance": {
        "source": "NFHS",
        "category": "Healthcare Access",
        "description": "Percentage of population with health insurance",
        "direction": "positive"
    },
    "Out_Of_Pocket_Expense": {
        "source": "NFHS",
        "category": "Healthcare Access",
        "description": "Out-of-pocket health expenditure",
        "direction": "negative"
    },
    "Clean_Fuel": {
        "source": "NFHS",
        "category": "Infrastructure",
        "description": "Percentage of households using clean fuel",
        "direction": "positive"
    },
    "Improved_Sanitation": {
        "source": "NFHS",
        "category": "Infrastructure",
        "description": "Percentage of households with improved sanitation",
        "direction": "positive"
    },
    "Improved_Drinking_Water": {
        "source": "NFHS",
        "category": "Infrastructure",
        "description": "Percentage of households with improved drinking water",
        "direction": "positive"
    },
    "Electricity": {
        "source": "NFHS",
        "category": "Infrastructure",
        "description": "Percentage of households with electricity",
        "direction": "positive"
    },
    "Women_10Plus_Schooling": {
        "source": "NFHS",
        "category": "Education",
        "description": "Percentage of women aged 10+ attending school",
        "direction": "positive"
    },
    "Men_Alcohol": {
        "source": "NFHS",
        "category": "Substance Use",
        "description": "Percentage of men consuming alcohol",
        "direction": "negative"
    },
    "Women_Alcohol": {
        "source": "NFHS",
        "category": "Substance Use",
        "description": "Percentage of women consuming alcohol",
        "direction": "negative"
    },
    "Men_Tobacco": {
        "source": "NFHS",
        "category": "Substance Use",
        "description": "Percentage of men using tobacco",
        "direction": "negative"
    },
    "Women_Tobacco": {
        "source": "NFHS",
        "category": "Substance Use",
        "description": "Percentage of women using tobacco",
        "direction": "negative"
    },
    "Children_Underweight": {
        "source": "NFHS",
        "category": "Nutrition",
        "description": "Percentage of underweight children",
        "direction": "negative"
    },
    "Children_Stunted": {
        "source": "NFHS",
        "category": "Nutrition",
        "description": "Percentage of stunted children",
        "direction": "negative"
    },
    "Children_Wasted": {
        "source": "NFHS",
        "category": "Nutrition",
        "description": "Percentage of wasted children",
        "direction": "negative"
    },
    "Women_Low_BMI": {
        "source": "NFHS",
        "category": "Nutrition",
        "description": "Percentage of women with low BMI",
        "direction": "negative"
    },
    "Women_Overweight": {
        "source": "NFHS",
        "category": "Nutrition",
        "description": "Percentage of overweight women",
        "direction": "negative"
    },
    "ANC_First_Trimester": {
        "source": "NFHS",
        "category": "Maternal Health",
        "description": "Percentage of mothers receiving ANC in first trimester",
        "direction": "positive"
    },
    "ANC_4Plus_Visits": {
        "source": "NFHS",
        "category": "Maternal Health",
        "description": "Percentage of mothers with 4+ ANC visits",
        "direction": "positive"
    },
    "IFA_100_Days": {
        "source": "NFHS",
        "category": "Maternal Health",
        "description": "Percentage of mothers receiving IFA for 100+ days",
        "direction": "positive"
    },
    "Breast_Cancer_Screening": {
        "source": "NFHS",
        "category": "Cancer Screening",
        "description": "Percentage of women screened for breast cancer",
        "direction": "positive"
    },
    "Cervical_Cancer_Screening": {
        "source": "NFHS",
        "category": "Cancer Screening",
        "description": "Percentage of women screened for cervical cancer",
        "direction": "positive"
    },
    "Oral_Cancer_Screening": {
        "source": "NFHS",
        "category": "Cancer Screening",
        "description": "Percentage of population screened for oral cancer",
        "direction": "positive"
    },
    "Functional_Sub_Centres": {
        "source": "RHS",
        "category": "Health Facilities",
        "description": "Number of functional sub-centres",
        "direction": "positive"
    },
    "Functional_PHCs": {
        "source": "RHS",
        "category": "Health Facilities",
        "description": "Number of functional primary health centres",
        "direction": "positive"
    },
    "Functional_CHCs": {
        "source": "RHS",
        "category": "Health Facilities",
        "description": "Number of functional community health centres",
        "direction": "positive"
    },
    "Functional_SDHs": {
        "source": "RHS",
        "category": "Health Facilities",
        "description": "Number of functional sub-district hospitals",
        "direction": "positive"
    },
    "Functional_DHs": {
        "source": "RHS",
        "category": "Health Facilities",
        "description": "Number of functional district hospitals",
        "direction": "positive"
    },
    "Landless_Manual_Labour": {
        "source": "SECC",
        "category": "Economic Status",
        "description": "Percentage of landless manual labourers",
        "direction": "negative"
    },
    "Non_Agricultural_Enterprise": {
        "source": "SECC",
        "category": "Economic Status",
        "description": "Number of non-agricultural enterprises",
        "direction": "positive"
    },
    "Income_Tax_Households": {
        "source": "SECC",
        "category": "Economic Status",
        "description": "Number of income tax households",
        "direction": "positive"
    },
    "Government_Salaried": {
        "source": "SECC",
        "category": "Economic Status",
        "description": "Percentage of government salaried workers",
        "direction": "positive"
    },
    "Private_Salaried": {
        "source": "SECC",
        "category": "Economic Status",
        "description": "Percentage of private salaried workers",
        "direction": "positive"
    },
    "Income_Above_10000": {
        "source": "SECC",
        "category": "Economic Status",
        "description": "Percentage of households with income above 10000",
        "direction": "positive"
    },
    "Refrigerator": {
        "source": "SECC",
        "category": "Asset Ownership",
        "description": "Percentage of households with refrigerator",
        "direction": "positive"
    },
    "Four_Wheeler": {
        "source": "SECC",
        "category": "Asset Ownership",
        "description": "Percentage of households with four-wheeler",
        "direction": "positive"
    },
    "Kisan_Credit_Card": {
        "source": "SECC",
        "category": "Economic Status",
        "description": "Number of Kisan Credit Card holders",
        "direction": "positive"
    },
    "Reporting_Offices": {
        "source": "RBI",
        "category": "Banking",
        "description": "Number of reporting offices",
        "direction": "positive"
    },
    "Aggregate_Deposit": {
        "source": "RBI",
        "category": "Banking",
        "description": "Total aggregate deposits",
        "direction": "positive"
    },
    "Gross_Bank_Credit": {
        "source": "RBI",
        "category": "Banking",
        "description": "Total gross bank credit",
        "direction": "positive"
    }
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_input_file(filepath: Path) -> None:
    """
    Validate that input file exists.
    
    Args:
        filepath: Path to input CSV file.
        
    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Validate that dataframe contains all required columns.
    
    Args:
        df: Input dataframe.
        required_cols: List of required column names.
        
    Raises:
        ValueError: If any required column is missing.
    """
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def validate_key_columns(df: pd.DataFrame, key_cols: List[str]) -> None:
    """
    Validate that key columns have no missing values.
    
    Args:
        df: Input dataframe.
        key_cols: List of key column names.
        
    Raises:
        ValueError: If key columns contain missing values.
    """
    for col in key_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            raise ValueError(
                f"Key column '{col}' contains {missing_count} missing values"
            )


def validate_duplicate_keys(df: pd.DataFrame, key_cols: List[str]) -> None:
    """
    Validate that no duplicate key combinations exist.
    
    Args:
        df: Input dataframe.
        key_cols: List of key column names.
        
    Raises:
        ValueError: If duplicate keys are found.
    """
    duplicates = df.duplicated(subset=key_cols).sum()
    
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate State-District pairs")


def run_validation(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Run all validation checks.
    
    Args:
        df: Input dataframe.
        logger: Logger instance.
    """
    logger.info("Starting validation checks...")
    
    try:
        validate_required_columns(df, KEY_COLUMNS + ALL_FEATURE_COLUMNS)
        logger.info("PASS All required columns present")
        
        validate_key_columns(df, KEY_COLUMNS)
        logger.info("PASS Key columns have no missing values")
        
        validate_duplicate_keys(df, KEY_COLUMNS)
        logger.info("PASS No duplicate State-District pairs")
        
        logger.info("Validation completed successfully")
        
        assert len(df) > 0, "Dataframe is empty"
        assert df["State"].isna().sum() == 0, "State column contains missing values"
        assert df["District"].isna().sum() == 0, "District column contains missing values"
        assert df.duplicated(subset=KEY_COLUMNS).sum() == 0, "Duplicate State-District pairs exist"
        
        logger.info("PASS All runtime assertions passed")
    
    except (FileNotFoundError, ValueError, AssertionError) as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ VALIDATION ERROR: {e}")
        sys.exit(1)


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def convert_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric, coercing errors to NaN.
    
    Args:
        df: Input dataframe.
        columns: List of columns to convert.
        
    Returns:
        Dataframe with converted columns.
    """
    df_copy = df.copy()
    
    for col in columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    
    return df_copy


def impute_missing_values(
    df: pd.DataFrame,
    columns: List[str],
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Impute missing values using median strategy.
    
    Args:
        df: Input dataframe.
        columns: List of columns to impute.
        logger: Logger instance.
        
    Returns:
        Tuple of imputed dataframe and imputation statistics.
    """
    df_copy = df.copy()
    imputation_stats = {}
    
    for col in columns:
        missing_before = df_copy[col].isna().sum()
        
        if missing_before > 0:
            median_value = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(median_value)
            
            imputation_stats[col] = {
                "missing_count": missing_before,
                "imputation_value": median_value
            }
            
            logger.debug(
                f"Imputed {missing_before} values in '{col}' "
                f"using median {median_value:.4f}"
            )
    
    return df_copy, imputation_stats


# =============================================================================
# QUALITY REPORT FUNCTIONS
# =============================================================================

def generate_data_quality_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feature_columns: List[str],
    report_path: Path,
    logger: logging.Logger
) -> None:
    """
    Generate comprehensive data quality report.
    
    Args:
        df_before: Dataframe before imputation.
        df_after: Dataframe after imputation.
        feature_columns: List of feature columns.
        report_path: Path to save report.
        logger: Logger instance.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATA QUALITY REPORT - PMAF FEATURE ENGINEERING\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET DIMENSIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Rows (Before)              : {len(df_before)}\n")
        f.write(f"Rows (After)               : {len(df_after)}\n")
        f.write(f"Columns (Before)           : {len(df_before.columns)}\n")
        f.write(f"Columns (After)            : {len(df_after.columns)}\n")
        f.write(f"Selected Features          : {len(feature_columns)}\n\n")
        
        f.write("MISSING VALUES SUMMARY\n")
        f.write("-" * 80 + "\n")
        
        missing_before = df_before[feature_columns].isna().sum().sum()
        missing_after = df_after[feature_columns].isna().sum().sum()
        
        f.write(f"Total Missing Values (Before) : {missing_before}\n")
        f.write(f"Total Missing Values (After)  : {missing_after}\n")
        f.write(f"Missing Values Imputed        : {missing_before - missing_after}\n\n")
        
        f.write("DUPLICATE KEY CHECK\n")
        f.write("-" * 80 + "\n")
        f.write(f"Duplicate State-District Pairs : "
                f"{df_after.duplicated(subset=['State', 'District']).sum()}\n\n")
        
        f.write("COLUMNS WITH HIGHEST MISSING VALUES (Before Imputation)\n")
        f.write("-" * 80 + "\n")
        
        missing_by_column = df_before[feature_columns].isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0].sort_values(
            ascending=False
        )
        
        if len(missing_by_column) == 0:
            f.write("No missing values found.\n\n")
        else:
            for col, count in missing_by_column.head(10).items():
                percentage = (count / len(df_before)) * 100
                f.write(f"{col:<40} : {count:>6} ({percentage:>6.2f}%)\n")
            f.write("\n")
        
        f.write("DATA TYPE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Numeric Columns : {len(df_after.select_dtypes(include=np.number).columns)}\n")
        f.write(f"Object Columns  : {len(df_after.select_dtypes(include=['object', 'str']).columns)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Data quality report saved to {report_path}")


def generate_missing_value_report(
    df: pd.DataFrame,
    feature_columns: List[str],
    report_path: Path,
    timing: str,
    logger: logging.Logger
) -> None:
    """
    Generate missing value report as CSV.
    
    Args:
        df: Input dataframe.
        feature_columns: List of feature columns.
        report_path: Path to save report.
        timing: Either "before" or "after" imputation.
        logger: Logger instance.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    missing_data = []
    
    for col in feature_columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        missing_data.append({
            "Feature": col,
            "Missing_Count": missing_count,
            "Missing_Percentage": round(missing_pct, 2),
            "Data_Type": str(df[col].dtype)
        })
    
    missing_df = pd.DataFrame(missing_data).sort_values(
        "Missing_Count",
        ascending=False
    )
    
    missing_df.to_csv(report_path, index=False)
    logger.info(f"Missing value report ({timing} imputation) saved to {report_path}")


def generate_feature_statistics(
    df: pd.DataFrame,
    feature_columns: List[str],
    report_path: Path,
    logger: logging.Logger
) -> None:
    """
    Generate descriptive statistics report.
    
    Args:
        df: Input dataframe.
        feature_columns: List of feature columns.
        report_path: Path to save report.
        logger: Logger instance.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = df[feature_columns].describe().T
    stats["Range"] = stats["max"] - stats["min"]
    
    stats.to_csv(report_path)
    logger.info(f"Feature statistics saved to {report_path}")


def generate_feature_dictionary(
    feature_columns: List[str],
    report_path: Path,
    logger: logging.Logger
) -> None:
    """
    Generate feature dictionary as CSV from FEATURE_METADATA.
    
    Args:
        feature_columns: List of feature columns.
        report_path: Path to save report.
        logger: Logger instance.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    dictionary_data = []
    
    for feat in feature_columns:
        metadata = FEATURE_METADATA.get(feat, {})
        dictionary_data.append({
            "Feature": feat,
            "Source_Dataset": metadata.get("source", "Unknown"),
            "Feature_Category": metadata.get("category", "Unknown"),
            "Description": metadata.get("description", ""),
            "Direction": metadata.get("direction", "neutral")
        })
    
    dict_df = pd.DataFrame(dictionary_data)
    dict_df.to_csv(report_path, index=False)
    logger.info(f"Feature dictionary saved to {report_path}")


def detect_outliers_iqr(
    df: pd.DataFrame,
    feature_columns: List[str],
    report_path: Path,
    logger: logging.Logger
) -> None:
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df: Input dataframe.
        feature_columns: List of feature columns.
        report_path: Path to save report.
        logger: Logger instance.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    outlier_data = []
    
    for col in feature_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
        
        outlier_data.append({
            "Feature": col,
            "Q1": round(Q1, 4),
            "Q3": round(Q3, 4),
            "IQR": round(IQR, 4),
            "Lower_Bound": round(lower_bound, 4),
            "Upper_Bound": round(upper_bound, 4),
            "Outlier_Count": outlier_count,
            "Outlier_Percentage": round(outlier_pct, 2)
        })
    
    outlier_df = pd.DataFrame(outlier_data).sort_values(
        "Outlier_Count",
        ascending=False
    )
    
    outlier_df.to_csv(report_path, index=False)
    logger.info(f"Outlier report saved to {report_path}")


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_feature_engineering_pipeline(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    reports_dir: Path,
    tables_dir: Path
) -> None:
    """
    Execute complete feature engineering pipeline.
    
    Args:
        input_path: Path to input CSV file.
        output_path: Path to save output CSV file.
        log_path: Path to save log file.
        reports_dir: Path to save reports.
        tables_dir: Path to save output tables.
    """
    start_time = time.time()
    
    logger = setup_logging(log_path)
    logger.info("=" * 80)
    logger.info("PMAF FEATURE ENGINEERING PIPELINE STARTED")
    logger.info("=" * 80)
    
    try:
        # Load data
        logger.info(f"Loading input data from {input_path}...")
        validate_input_file(input_path)
        df_raw = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df_raw)} rows and {len(df_raw.columns)} columns")
        
        # Validate
        logger.info("Running validation checks...")
        run_validation(df_raw, logger)
        
        # Clean and prepare
        logger.info("Converting feature columns to numeric...")
        df_cleaned = convert_to_numeric(df_raw, ALL_FEATURE_COLUMNS)
        
        logger.info("Imputing missing values using median strategy...")
        df_imputed, imputation_stats = impute_missing_values(
            df_cleaned,
            ALL_FEATURE_COLUMNS,
            logger
        )
        
        # Generate reports
        logger.info("Generating data quality report...")
        generate_data_quality_report(
            df_cleaned,
            df_imputed,
            ALL_FEATURE_COLUMNS,
            reports_dir / "data_quality_report.txt",
            logger
        )
        
        logger.info("Generating missing value report (before imputation)...")
        generate_missing_value_report(
            df_cleaned,
            ALL_FEATURE_COLUMNS,
            tables_dir / "missing_value_report_before.csv",
            "before",
            logger
        )
        
        logger.info("Generating missing value report (after imputation)...")
        generate_missing_value_report(
            df_imputed,
            ALL_FEATURE_COLUMNS,
            tables_dir / "missing_value_report_after.csv",
            "after",
            logger
        )
        
        logger.info("Generating feature statistics...")
        generate_feature_statistics(
            df_imputed,
            ALL_FEATURE_COLUMNS,
            tables_dir / "feature_statistics.csv",
            logger
        )
        
        logger.info("Generating feature dictionary...")
        generate_feature_dictionary(
            ALL_FEATURE_COLUMNS,
            tables_dir / "feature_dictionary.csv",
            logger
        )
        
        logger.info("Detecting outliers...")
        detect_outliers_iqr(
            df_imputed,
            ALL_FEATURE_COLUMNS,
            tables_dir / "outlier_report.csv",
            logger
        )
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Selecting engineered features...")
        df_final = df_imputed[KEY_COLUMNS + ALL_FEATURE_COLUMNS].copy()
        logger.info(f"Saving engineered features to {output_path}...")
        df_final.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        
        # Print execution summary
        print_execution_summary(
            len(df_raw),
            len(df_raw.columns),
            len(ALL_FEATURE_COLUMNS),
            imputation_stats,
            elapsed_time
        )
        
        logger.info("=" * 80)
        logger.info("PMAF FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ PIPELINE ERROR: {e}")
        sys.exit(1)


def print_execution_summary(
    input_rows: int,
    input_columns: int,
    selected_features: int,
    imputation_stats: Dict,
    elapsed_time: float
) -> None:
    """
    Print professional execution summary.
    
    Args:
        input_rows: Number of input rows.
        input_columns: Number of input columns.
        selected_features: Number of selected features.
        imputation_stats: Dictionary of imputation statistics.
        elapsed_time: Total execution time in seconds.
    """
    total_imputed = sum(
        stat["missing_count"] for stat in imputation_stats.values()
    )
    
    print("\n")
    print("=" * 80)
    print("PMAF FEATURE ENGINEERING")
    print("=" * 80)
    print()
    print(f"Input Dataset")
    print(f"  {input_rows} rows")
    print(f"  {input_columns} columns")
    print()
    print(f"↓")
    print()
    print(f"Selected Features")
    print(f"  {selected_features} columns")
    print()
    print(f"↓")
    print()
    print(f"Missing Values Filled")
    print(f"  {total_imputed} imputed (median strategy)")
    print()
    print(f"↓")
    print()
    print(f"Reports Generated")
    print(f"  • feature_statistics.csv")
    print(f"  • missing_value_report.csv")
    print(f"  • feature_dictionary.csv")
    print(f"  • outlier_report.csv")
    print(f"  • data_quality_report.txt")
    print()
    print(f"↓")
    print()
    print(f"Output Saved")
    print(f"  data/final/pmaf_features.csv")
    print()
    print(f"Execution Time")
    print(f"  {elapsed_time:.2f} seconds")
    print()
    print("=" * 80)
    print("✓ SUCCESS")
    print("=" * 80)
    print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    INPUT_PATH = Path("data/merged/merged_dataset.csv")
    OUTPUT_PATH = Path("data/final/pmaf_features.csv")
    LOG_PATH = Path("outputs/reports/feature_engineering.log")
    REPORTS_DIR = Path("outputs/reports")
    TABLES_DIR = Path("outputs/tables")
    
    run_feature_engineering_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        log_path=LOG_PATH,
        reports_dir=REPORTS_DIR,
        tables_dir=TABLES_DIR
    )
