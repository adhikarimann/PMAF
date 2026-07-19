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
    
    logger = logging.getLogger("PMAF_Score")
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

IDENTIFIER_COLUMNS: List[str] = ["State", "District"]

MARKET_SIZE_FEATURES: List[str] = [
    "Population",
    "Households_x"
]

CHRONIC_DISEASE_FEATURES: List[str] = [
    "Women_Anaemia",
    "Pregnant_Anaemia",
    "Children_Anaemia",
    "Men_High_BloodSugar",
    "Women_High_BloodSugar",
    "Men_Elevated_BP",
    "Women_Elevated_BP",
    "Men_Tobacco",
    "Women_Tobacco",
    "Men_Alcohol",
    "Women_Alcohol",
    "Women_Overweight"
]

ACUTE_DISEASE_FEATURES: List[str] = [
    "ARI_Prevalence",
    "Diarrhoea_Prevalence",
    "Children_Underweight",
    "Children_Stunted",
    "Children_Wasted",
    "Women_Low_BMI"
]

HEALTHCARE_FEATURES: List[str] = [
    "Functional_Sub_Centres",
    "Functional_PHCs",
    "Functional_CHCs",
    "Functional_SDHs",
    "Functional_DHs",
    "Health_Insurance",
    "Skilled_Births",
    "Institutional_Births",
    "ANC_First_Trimester",
    "ANC_4Plus_Visits",
    "IFA_100_Days",
    "Breast_Cancer_Screening",
    "Cervical_Cancer_Screening",
    "Oral_Cancer_Screening"
]

ECONOMIC_FEATURES: List[str] = [
    "Income_Tax_Households",
    "Government_Salaried",
    "Private_Salaried",
    "Income_Above_10000",
    "Refrigerator",
    "Four_Wheeler",
    "Aggregate_Deposit",
    "Gross_Bank_Credit",
    "Reporting_Offices",
    "Kisan_Credit_Card",
    "Landless_Manual_Labour"
]

DEVELOPMENT_FEATURES: List[str] = [
    "Literacy_Rate",
    "Women_10Plus_Schooling",
    "Electricity",
    "Clean_Fuel",
    "Improved_Sanitation",
    "Improved_Drinking_Water",
    "Non_Agricultural_Enterprise",
    "Out_Of_Pocket_Expense"
]

ALL_FEATURE_COLUMNS = (
    MARKET_SIZE_FEATURES +
    CHRONIC_DISEASE_FEATURES +
    ACUTE_DISEASE_FEATURES +
    HEALTHCARE_FEATURES +
    ECONOMIC_FEATURES +
    DEVELOPMENT_FEATURES
)

PILLAR_SCORE_COLUMNS: List[str] = [
    "Market_Size_Score",
    "Chronic_Disease_Score",
    "Acute_Disease_Score",
    "Healthcare_Score",
    "Economic_Score",
    "Development_Score"
]

PMAF_SCORE_COLUMNS: List[str] = [
    "Overall_PMAF",
    "Chronic_PMAF",
    "Acute_PMAF"
]

# =============================================================================
# WEIGHT DEFINITIONS
# =============================================================================

OVERALL_WEIGHTS: Dict[str, float] = {
    "Market_Size_Score": 0.20,
    "Chronic_Disease_Score": 0.10,
    "Acute_Disease_Score": 0.10,
    "Healthcare_Score": 0.20,
    "Economic_Score": 0.20,
    "Development_Score": 0.20
}

CHRONIC_WEIGHTS: Dict[str, float] = {
    "Market_Size_Score": 0.20,
    "Chronic_Disease_Score": 0.35,
    "Acute_Disease_Score": 0.00,
    "Healthcare_Score": 0.20,
    "Economic_Score": 0.15,
    "Development_Score": 0.10
}

ACUTE_WEIGHTS: Dict[str, float] = {
    "Market_Size_Score": 0.30,
    "Chronic_Disease_Score": 0.00,
    "Acute_Disease_Score": 0.25,
    "Healthcare_Score": 0.20,
    "Economic_Score": 0.10,
    "Development_Score": 0.15
}


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def load_dataset(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load normalized features dataset.
    
    Args:
        input_path: Path to input CSV file.
        logger: Logger instance.
        
    Returns:
        Loaded dataframe.
        
    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def validate_dataset(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Validate dataset integrity before scoring.
    
    Args:
        df: Input dataframe.
        logger: Logger instance.
        
    Raises:
        ValueError: If validation fails.
        AssertionError: If assertions fail.
    """
    logger.info("Starting dataset validation...")
    
    try:
        # Check identifiers exist
        for col in IDENTIFIER_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Required identifier column missing: {col}")
        logger.info("OK Identifier columns present")
        
        # Check required features exist
        for col in ALL_FEATURE_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Required feature column missing: {col}")
        logger.info(f"OK All {len(ALL_FEATURE_COLUMNS)} feature columns present")
        
        # Check for missing identifiers
        for col in IDENTIFIER_COLUMNS:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                raise ValueError(
                    f"Identifier column '{col}' contains {missing_count} missing values"
                )
        logger.info("OK No missing identifier values")
        
        # Check for duplicate keys
        duplicates = df.duplicated(subset=IDENTIFIER_COLUMNS).sum()
        if duplicates > 0:
            raise ValueError(f"Found {duplicates} duplicate State-District pairs")
        logger.info("OK No duplicate State-District pairs")
        
        # Check for missing values in features
        feature_missing = df[ALL_FEATURE_COLUMNS].isna().sum().sum()
        if feature_missing > 0:
            raise ValueError(
                f"Found {feature_missing} missing values in feature columns"
            )
        logger.info("OK No missing values in feature columns")
        
        # Check feature bounds [0, 1]
        for col in ALL_FEATURE_COLUMNS:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < -1e-6 or max_val > 1 + 1e-6:
                raise ValueError(
                    f"Feature '{col}' out of bounds: [{min_val:.6f}, {max_val:.6f}]"
                )
        logger.info("OK All features within [0, 1] bounds")
        
        # Runtime assertions
        assert len(df) > 0, "Dataframe is empty"
        assert df["State"].notna().all(), "State contains missing values"
        assert df["District"].notna().all(), "District contains missing values"
        assert not df.duplicated(subset=IDENTIFIER_COLUMNS).any(), "Duplicate keys exist"
        assert not df[ALL_FEATURE_COLUMNS].isna().any().any(), "NaN in features"
        
        logger.info("OK All validation checks passed")
    
    except (ValueError, AssertionError) as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ VALIDATION ERROR: {e}")
        sys.exit(1)


# =============================================================================
# PILLAR SCORE CALCULATION
# =============================================================================

def calculate_pillar_score(
    df: pd.DataFrame,
    features: List[str],
    pillar_name: str,
    logger: logging.Logger
) -> pd.Series:
    """
    Calculate pillar score as arithmetic mean of features.
    
    Args:
        df: Input dataframe.
        features: List of feature columns for this pillar.
        pillar_name: Name of the pillar for logging.
        logger: Logger instance.
        
    Returns:
        Series of pillar scores (values 0-1).
    """
    logger.debug(f"Calculating {pillar_name} from {len(features)} features")
    
    pillar_score = df[features].mean(axis=1)
    
    assert pillar_score.notna().all(), f"{pillar_name} contains NaN values"
    assert (pillar_score >= -1e-6).all() and (pillar_score <= 1 + 1e-6).all(), \
        f"{pillar_name} out of bounds"
    
    logger.debug(
        f"  Mean: {pillar_score.mean():.4f}, "
        f"Min: {pillar_score.min():.4f}, "
        f"Max: {pillar_score.max():.4f}"
    )
    
    return pillar_score


def calculate_all_pillar_scores(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Calculate all six pillar scores.
    
    Args:
        df: Input dataframe with normalized features.
        logger: Logger instance.
        
    Returns:
        Dataframe with pillar scores added.
    """
    logger.info("Calculating pillar scores...")
    
    df_scores = df.copy()
    
    df_scores["Market_Size_Score"] = calculate_pillar_score(
        df, MARKET_SIZE_FEATURES, "Market_Size", logger
    )
    
    df_scores["Chronic_Disease_Score"] = calculate_pillar_score(
        df, CHRONIC_DISEASE_FEATURES, "Chronic_Disease", logger
    )
    
    df_scores["Acute_Disease_Score"] = calculate_pillar_score(
        df, ACUTE_DISEASE_FEATURES, "Acute_Disease", logger
    )
    
    df_scores["Healthcare_Score"] = calculate_pillar_score(
        df, HEALTHCARE_FEATURES, "Healthcare", logger
    )
    
    df_scores["Economic_Score"] = calculate_pillar_score(
        df, ECONOMIC_FEATURES, "Economic", logger
    )
    
    df_scores["Development_Score"] = calculate_pillar_score(
        df, DEVELOPMENT_FEATURES, "Development", logger
    )
    
    logger.info("OK All pillar scores calculated")
    
    return df_scores


# =============================================================================
# PMAF SCORE CALCULATION
# =============================================================================

def calculate_pmaf_score(
    df: pd.DataFrame,
    weights: Dict[str, float],
    score_name: str,
    logger: logging.Logger
) -> pd.Series:
    """
    Calculate PMAF score as weighted sum of pillar scores.
    
    Args:
        df: Dataframe with pillar scores.
        weights: Dictionary of weights for each pillar.
        score_name: Name of the PMAF score for logging.
        logger: Logger instance.
        
    Returns:
        Series of PMAF scores (values 0-100).
    """
    logger.debug(f"Calculating {score_name}...")
    
    pmaf_score = pd.Series(0.0, index=df.index)
    
    for pillar, weight in weights.items():
        pmaf_score += df[pillar] * weight
    
    pmaf_score = pmaf_score * 100
    
    assert pmaf_score.notna().all(), f"{score_name} contains NaN values"
    assert (pmaf_score >= -1e-3).all() and (pmaf_score <= 100 + 1e-3).all(), \
        f"{score_name} out of bounds [0, 100]"
    
    logger.debug(
        f"  Mean: {pmaf_score.mean():.2f}, "
        f"Min: {pmaf_score.min():.2f}, "
        f"Max: {pmaf_score.max():.2f}"
    )
    
    return pmaf_score


def calculate_all_pmaf_scores(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Calculate all three PMAF scores.
    
    Args:
        df: Dataframe with pillar scores.
        logger: Logger instance.
        
    Returns:
        Dataframe with PMAF scores added.
    """
    logger.info("Calculating PMAF scores...")
    
    df_pmaf = df.copy()
    
    df_pmaf["Overall_PMAF"] = calculate_pmaf_score(
        df, OVERALL_WEIGHTS, "Overall_PMAF", logger
    )
    
    df_pmaf["Chronic_PMAF"] = calculate_pmaf_score(
        df, CHRONIC_WEIGHTS, "Chronic_PMAF", logger
    )
    
    df_pmaf["Acute_PMAF"] = calculate_pmaf_score(
        df, ACUTE_WEIGHTS, "Acute_PMAF", logger
    )
    
    logger.info("OK All PMAF scores calculated")
    
    return df_pmaf


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_score_summary(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    """
    Generate summary statistics for all scores.
    
    Args:
        df: Dataframe with calculated scores.
        logger: Logger instance.
        
    Returns:
        Dictionary of summary statistics.
    """
    logger.debug("Generating score summary...")
    
    summary = {}
    all_score_columns = PILLAR_SCORE_COLUMNS + PMAF_SCORE_COLUMNS
    
    for col in all_score_columns:
        summary[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }
    
    return summary


def print_execution_summary(
    input_rows: int,
    execution_time: float,
    summary_stats: Dict[str, Dict[str, float]],
    logger: logging.Logger
) -> None:
    """
    Print professional execution summary to console.
    
    Args:
        input_rows: Number of districts processed.
        execution_time: Total execution time in seconds.
        summary_stats: Summary statistics dictionary.
        logger: Logger instance.
    """
    print("\n")
    print("=" * 80)
    print("PMAF SCORE CALCULATION")
    print("=" * 80)
    print()
    print(f"Input Dataset")
    print(f"  {input_rows} districts")
    print()
    print(f"↓")
    print()
    print(f"Pillar Scores Calculated")
    print(f"  Market Size Score")
    print(f"  Chronic Disease Score")
    print(f"  Acute Disease Score")
    print(f"  Healthcare Score")
    print(f"  Economic Score")
    print(f"  Development Score")
    print()
    print(f"↓")
    print()
    print(f"PMAF Scores Calculated")
    print(f"  Overall PMAF")
    print(f"  Chronic PMAF")
    print(f"  Acute PMAF")
    print()
    print(f"↓")
    print()
    print(f"Validation Passed")
    print(f"  OK No missing values")
    print(f"  OK No duplicates")
    print(f"  OK All pillar scores ∈ [0, 1]")
    print(f"  OK All PMAF scores ∈ [0, 100]")
    print()
    print(f"↓")
    print()
    print(f"Output Saved")
    print(f"  data/final/pmaf_scores.csv")
    print()
    print(f"Execution Time")
    print(f"  {execution_time:.2f} seconds")
    print()
    print("=" * 80)
    print("OK SUCCESS")
    print("=" * 80)
    print()
    
    logger.info("Execution summary printed to console")


# =============================================================================
# SAVE OUTPUT
# =============================================================================

def save_output_dataset(
    df: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """
    Save calculated scores to CSV.
    
    Args:
        df: Dataframe with all scores.
        output_path: Path to save output.
        logger: Logger instance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_columns = (
        IDENTIFIER_COLUMNS +
        PILLAR_SCORE_COLUMNS +
        PMAF_SCORE_COLUMNS
    )
    
    df_output = df[output_columns].copy()
    
    logger.info(f"Saving results to {output_path}...")
    df_output.to_csv(output_path, index=False)
    
    logger.info(f"Output saved successfully")
    logger.info(f"Output shape: {df_output.shape[0]} rows × {df_output.shape[1]} columns")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Execute complete PMAF score calculation pipeline.
    """
    start_time = time.time()
    
    input_path = Path("data/final/normalized_features.csv")
    output_path = Path("data/final/pmaf_scores.csv")
    log_path = Path("outputs/reports/pmaf_score_calculation.log")
    
    logger = setup_logging(log_path)
    
    logger.info("=" * 80)
    logger.info("PMAF SCORE CALCULATION PIPELINE STARTED")
    logger.info("=" * 80)
    
    try:
        # Load dataset
        df_input = load_dataset(input_path, logger)
        
        # Validate
        validate_dataset(df_input, logger)
        
        # Calculate pillar scores
        df_pillars = calculate_all_pillar_scores(df_input, logger)
        
        # Calculate PMAF scores
        df_final = calculate_all_pmaf_scores(df_pillars, logger)
        
        # Generate summary
        summary_stats = generate_score_summary(df_final, logger)
        
        # Save output
        save_output_dataset(df_final, output_path, logger)
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print_execution_summary(
            len(df_input),
            elapsed_time,
            summary_stats,
            logger
        )
        
        logger.info("=" * 80)
        logger.info("PMAF SCORE CALCULATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ PIPELINE ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()