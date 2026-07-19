import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import List, Tuple
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
    
    logger = logging.getLogger("PMAF_Ranking")
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
# COLUMN DEFINITIONS
# =============================================================================

IDENTIFIER_COLUMNS: List[str] = ["State", "District"]

SCORE_COLUMNS: List[str] = [
    "Market_Size_Score",
    "Chronic_Disease_Score",
    "Acute_Disease_Score",
    "Healthcare_Score",
    "Economic_Score",
    "Development_Score",
    "Overall_PMAF",
    "Chronic_PMAF",
    "Acute_PMAF"
]

PMAF_SCORE_COLUMNS: List[str] = [
    "Overall_PMAF",
    "Chronic_PMAF",
    "Acute_PMAF"
]

RANK_COLUMNS: List[str] = [
    "Overall_Rank",
    "Chronic_Rank",
    "Acute_Rank"
]

OUTPUT_COLUMNS: List[str] = (
    IDENTIFIER_COLUMNS +
    SCORE_COLUMNS +
    RANK_COLUMNS
)

TOP_N_VALUES: List[int] = [10, 50, 100]


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def load_dataset(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load PMAF scores dataset.
    
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
    Validate dataset integrity before ranking.
    
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
        
        # Check score columns exist
        for col in SCORE_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Required score column missing: {col}")
        logger.info(f"OK All {len(SCORE_COLUMNS)} score columns present")
        
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
        
        # Check for missing scores
        score_missing = df[SCORE_COLUMNS].isna().sum().sum()
        if score_missing > 0:
            raise ValueError(f"Found {score_missing} missing values in score columns")
        logger.info("OK No missing score values")
        
        # Check PMAF score bounds [0, 100]
        for col in PMAF_SCORE_COLUMNS:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < -1e-3 or max_val > 100 + 1e-3:
                raise ValueError(
                    f"PMAF score '{col}' out of bounds: [{min_val:.4f}, {max_val:.4f}]"
                )
        logger.info("OK All PMAF scores within [0, 100] bounds")
        
        # Runtime assertions
        assert len(df) > 0, "Dataframe is empty"
        assert df["State"].notna().all(), "State contains missing values"
        assert df["District"].notna().all(), "District contains missing values"
        assert not df.duplicated(subset=IDENTIFIER_COLUMNS).any(), "Duplicate keys exist"
        assert not df[SCORE_COLUMNS].isna().any().any(), "NaN in score columns"
        
        logger.info("OK All validation checks passed")
    
    except (ValueError, AssertionError) as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n{chr(10)}{chr(10)}VALIDATION ERROR: {e}")
        sys.exit(1)


# =============================================================================
# RANKING FUNCTIONS
# =============================================================================

def calculate_rank(
    df: pd.DataFrame,
    score_column: str,
    rank_column: str,
    logger: logging.Logger
) -> pd.Series:
    """
    Calculate rank for a single PMAF score.
    
    Uses method='min' to handle ties (1,2,2,4 style).
    
    Args:
        df: Input dataframe.
        score_column: Name of the score column to rank.
        rank_column: Name of the output rank column.
        logger: Logger instance.
        
    Returns:
        Series of ranks.
    """
    logger.debug(f"Calculating {rank_column} from {score_column}...")
    
    rank = df[score_column].rank(method="min", ascending=False).astype(int)
    
    assert rank.notna().all(), f"{rank_column} contains NaN values"
    assert (rank >= 1).all(), f"{rank_column} contains invalid values"
    
    logger.debug(f"  {rank_column}: Min={rank.min()}, Max={rank.max()}")
    
    return rank


def calculate_all_ranks(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Calculate all ranking columns.
    
    Args:
        df: Input dataframe with PMAF scores.
        logger: Logger instance.
        
    Returns:
        Dataframe with ranking columns added.
    """
    logger.info("Calculating rankings...")
    
    df_ranked = df.copy()
    
    df_ranked["Overall_Rank"] = calculate_rank(
        df, "Overall_PMAF", "Overall_Rank", logger
    )
    
    df_ranked["Chronic_Rank"] = calculate_rank(
        df, "Chronic_PMAF", "Chronic_Rank", logger
    )
    
    df_ranked["Acute_Rank"] = calculate_rank(
        df, "Acute_PMAF", "Acute_Rank", logger
    )
    
    logger.info("OK All rankings calculated")
    
    return df_ranked


# =============================================================================
# TOP-N RANKING LISTS
# =============================================================================

def generate_top_n_list(
    df: pd.DataFrame,
    score_column: str,
    n: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Generate top-N list sorted by score.
    
    Args:
        df: Input dataframe with all columns.
        score_column: Column name to sort by.
        n: Number of top entries to return.
        logger: Logger instance.
        
    Returns:
        Dataframe containing top-N districts.
    """
    top_n = df.nlargest(n, score_column)[OUTPUT_COLUMNS].copy()
    
    logger.debug(f"Generated top-{n} list for {score_column}")
    
    return top_n


def generate_all_top_lists(
    df: pd.DataFrame,
    logger: logging.Logger
) -> dict:
    """
    Generate all top-N lists for all three PMAF scores.
    
    Args:
        df: Input dataframe with rankings.
        logger: Logger instance.
        
    Returns:
        Dictionary of top-N dataframes.
    """
    logger.info("Generating top-N lists...")
    
    top_lists = {}
    
    for n in TOP_N_VALUES:
        top_lists[f"top_{n}_overall"] = generate_top_n_list(
            df, "Overall_PMAF", n, logger
        )
        top_lists[f"top_{n}_chronic"] = generate_top_n_list(
            df, "Chronic_PMAF", n, logger
        )
        top_lists[f"top_{n}_acute"] = generate_top_n_list(
            df, "Acute_PMAF", n, logger
        )
    
    logger.info("OK All top-N lists generated")
    
    return top_lists


# =============================================================================
# FULL SORTED RANKING LISTS
# =============================================================================

def generate_full_sorted_list(
    df: pd.DataFrame,
    score_column: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Generate full sorted list by score.
    
    Args:
        df: Input dataframe with all columns.
        score_column: Column name to sort by.
        logger: Logger instance.
        
    Returns:
        Dataframe sorted descending by score.
    """
    sorted_df = df[OUTPUT_COLUMNS].sort_values(
        by=score_column,
        ascending=False
    ).reset_index(drop=True)
    
    logger.debug(f"Generated full sorted list for {score_column}")
    
    return sorted_df


def generate_all_sorted_lists(
    df: pd.DataFrame,
    logger: logging.Logger
) -> dict:
    """
    Generate full sorted lists for all three PMAF scores.
    
    Args:
        df: Input dataframe with rankings.
        logger: Logger instance.
        
    Returns:
        Dictionary of sorted dataframes.
    """
    logger.info("Generating full sorted ranking lists...")
    
    sorted_lists = {}
    
    sorted_lists["overall_ranking"] = generate_full_sorted_list(
        df, "Overall_PMAF", logger
    )
    sorted_lists["chronic_ranking"] = generate_full_sorted_list(
        df, "Chronic_PMAF", logger
    )
    sorted_lists["acute_ranking"] = generate_full_sorted_list(
        df, "Acute_PMAF", logger
    )
    
    logger.info("OK All full sorted lists generated")
    
    return sorted_lists


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_main_output(
    df: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """
    Save main ranked dataset.
    
    Args:
        df: Dataframe with rankings.
        output_path: Path to save output.
        logger: Logger instance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_output = df[OUTPUT_COLUMNS].copy()
    
    logger.info(f"Saving main output to {output_path}...")
    df_output.to_csv(output_path, index=False)
    
    logger.info(f"Main output saved successfully")
    logger.info(f"Shape: {df_output.shape[0]} rows x {df_output.shape[1]} columns")


def save_top_n_lists(
    top_lists: dict,
    rankings_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save all top-N list CSV files.
    
    Args:
        top_lists: Dictionary of top-N dataframes.
        rankings_dir: Directory to save files.
        logger: Logger instance.
    """
    rankings_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving top-N list files to {rankings_dir}...")
    
    for name, df in top_lists.items():
        file_path = rankings_dir / f"{name}.csv"
        df.to_csv(file_path, index=False)
        logger.debug(f"Saved {file_path}")
    
    logger.info(f"OK All top-N lists saved")


def save_sorted_lists(
    sorted_lists: dict,
    rankings_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save all full sorted ranking list CSV files.
    
    Args:
        sorted_lists: Dictionary of sorted dataframes.
        rankings_dir: Directory to save files.
        logger: Logger instance.
    """
    rankings_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving full sorted ranking files to {rankings_dir}...")
    
    for name, df in sorted_lists.items():
        file_path = rankings_dir / f"{name}.csv"
        df.to_csv(file_path, index=False)
        logger.debug(f"Saved {file_path}")
    
    logger.info(f"OK All sorted ranking files saved")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_execution_summary(
    input_rows: int,
    execution_time: float,
    logger: logging.Logger
) -> None:
    """
    Print professional execution summary to console.
    
    Args:
        input_rows: Number of districts ranked.
        execution_time: Total execution time in seconds.
        logger: Logger instance.
    """
    print("\n")
    print("=" * 80)
    print("DISTRICT RANKING")
    print("=" * 80)
    print()
    print(f"Input Dataset")
    print(f"  {input_rows} districts")
    print()
    print(f"->")
    print()
    print(f"Rankings Generated")
    print(f"  Overall Rank")
    print(f"  Chronic Rank")
    print(f"  Acute Rank")
    print()
    print(f"->")
    print()
    print(f"Output Saved")
    print(f"  district_rankings.csv")
    print()
    print(f"->")
    print()
    print(f"Top Lists Generated")
    print(f"  Top 10, Top 50, Top 100")
    print(f"  For each PMAF type (Overall, Chronic, Acute)")
    print()
    print(f"->")
    print()
    print(f"Full Sorted Rankings Generated")
    print(f"  overall_ranking.csv")
    print(f"  chronic_ranking.csv")
    print(f"  acute_ranking.csv")
    print()
    print(f"->")
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
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Execute complete district ranking pipeline.
    """
    start_time = time.time()
    
    input_path = Path("data/final/pmaf_scores.csv")
    output_path = Path("data/final/district_rankings.csv")
    rankings_dir = Path("outputs/rankings")
    log_path = Path("outputs/reports/district_ranking.log")
    
    logger = setup_logging(log_path)
    
    logger.info("=" * 80)
    logger.info("DISTRICT RANKING PIPELINE STARTED")
    logger.info("=" * 80)
    
    try:
        # Load dataset
        df_input = load_dataset(input_path, logger)
        
        # Validate
        validate_dataset(df_input, logger)
        
        # Calculate rankings
        df_ranked = calculate_all_ranks(df_input, logger)
        
        # Generate top-N lists
        top_lists = generate_all_top_lists(df_ranked, logger)
        
        # Generate full sorted lists
        sorted_lists = generate_all_sorted_lists(df_ranked, logger)
        
        # Save outputs
        save_main_output(df_ranked, output_path, logger)
        save_top_n_lists(top_lists, rankings_dir, logger)
        save_sorted_lists(sorted_lists, rankings_dir, logger)
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print_execution_summary(len(df_input), elapsed_time, logger)
        
        logger.info("=" * 80)
        logger.info("DISTRICT RANKING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n{chr(10)}{chr(10)}PIPELINE ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()