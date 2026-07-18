import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import time
import warnings


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
    
    logger = logging.getLogger("PMAF_Normalization")
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
# FEATURE CONFIGURATION
# =============================================================================

IDENTIFIER_COLUMNS: List[str] = ["State", "District"]

REVERSE_NORMALIZED_FEATURES: List[str] = [
    "Landless_Manual_Labour"
]


# =============================================================================
# DATASET LOADING AND VALIDATION
# =============================================================================

def load_dataset(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load engineered features dataset.
    
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
    Validate dataset integrity before normalization.
    
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
        logger.info("[OK] Identifier columns present")
        
        # Check for missing identifiers
        for col in IDENTIFIER_COLUMNS:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                raise ValueError(
                    f"Identifier column '{col}' contains {missing_count} missing values"
                )
        logger.info("[OK] No missing identifier values")
        
        # Check for duplicate keys
        duplicates = df.duplicated(subset=IDENTIFIER_COLUMNS).sum()
        if duplicates > 0:
            raise ValueError(f"Found {duplicates} duplicate State-District pairs")
        logger.info("[OK] No duplicate State-District pairs")
        
        # Runtime assertions
        assert len(df) > 0, "Dataframe is empty"
        assert df["State"].notna().all(), "State column contains missing values"
        assert df["District"].notna().all(), "District column contains missing values"
        assert not df.duplicated(subset=IDENTIFIER_COLUMNS).any(), "Duplicate keys exist"
        
        logger.info("[OK] All validation checks passed")
    
    except (ValueError, AssertionError) as e:
        logger.error(f"Validation failed: {e}")
        print(f"\nERROR VALIDATION ERROR: {e}")
        sys.exit(1)


def identify_numeric_features(
    df: pd.DataFrame,
    logger: logging.Logger
) -> List[str]:
    """
    Identify numeric features for normalization (exclude identifiers).
    
    Args:
        df: Input dataframe.
        logger: Logger instance.
        
    Returns:
        List of numeric feature column names.
    """
    all_columns = set(df.columns)
    identifier_set = set(IDENTIFIER_COLUMNS)
    feature_columns = all_columns - identifier_set
    
    numeric_features = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
    
    numeric_features.sort()
    logger.info(f"Identified {len(numeric_features)} numeric features for normalization")
    
    return numeric_features


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_feature_standard(series: pd.Series) -> pd.Series:
    """
    Apply standard Min-Max normalization: (x - min) / (max - min).
    
    Args:
        series: Input numeric series.
        
    Returns:
        Normalized series (values between 0 and 1).
    """
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(np.zeros_like(series), index=series.index, dtype=float)
    
    return (series - min_val) / (max_val - min_val)


def normalize_feature_reverse(series: pd.Series) -> pd.Series:
    """
    Apply reverse Min-Max normalization: (max - x) / (max - min).
    
    Used for features where lower values are more attractive (e.g., poverty).
    
    Args:
        series: Input numeric series.
        
    Returns:
        Normalized series (values between 0 and 1).
    """
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(np.zeros_like(series), index=series.index, dtype=float)
    
    return (max_val - series) / (max_val - min_val)


def normalize_features(
    df: pd.DataFrame,
    numeric_features: List[str],
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize all numeric features using Min-Max normalization.
    
    Standard normalization for most features.
    Reverse normalization for Landless_Manual_Labour only.
    
    Args:
        df: Input dataframe.
        numeric_features: List of numeric feature columns.
        logger: Logger instance.
        
    Returns:
        Tuple of normalized dataframe and normalization metadata.
    """
    df_normalized = df.copy()
    normalization_metadata = {}
    
    for feature in numeric_features:
        original_min = df[feature].min()
        original_max = df[feature].max()
        
        if feature in REVERSE_NORMALIZED_FEATURES:
            df_normalized[feature] = normalize_feature_reverse(df[feature])
            method = "Reverse Min-Max"
            logger.debug(
                f"Applied reverse normalization to '{feature}' "
                f"(min={original_min:.4f}, max={original_max:.4f})"
            )
        else:
            df_normalized[feature] = normalize_feature_standard(df[feature])
            method = "Standard Min-Max"
            logger.debug(
                f"Applied standard normalization to '{feature}' "
                f"(min={original_min:.4f}, max={original_max:.4f})"
            )
        
        normalized_min = df_normalized[feature].min()
        normalized_max = df_normalized[feature].max()
        normalized_mean = df_normalized[feature].mean()
        normalized_std = df_normalized[feature].std()
        
        normalization_metadata[feature] = {
            "original_min": original_min,
            "original_max": original_max,
            "normalized_min": normalized_min,
            "normalized_max": normalized_max,
            "normalized_mean": normalized_mean,
            "normalized_std": normalized_std,
            "method": method
        }
    
    logger.info(f"Normalization applied to {len(numeric_features)} features")
    return df_normalized, normalization_metadata


# =============================================================================
# VALIDATION AFTER NORMALIZATION
# =============================================================================

def validate_normalization(
    df: pd.DataFrame,
    numeric_features: List[str],
    logger: logging.Logger
) -> None:
    """
    Validate that normalization was applied correctly.
    
    Args:
        df: Normalized dataframe.
        numeric_features: List of normalized feature columns.
        logger: Logger instance.
        
    Raises:
        ValueError: If validation fails.
        AssertionError: If assertions fail.
    """
    logger.info("Starting normalization validation...")
    
    try:
        for feature in numeric_features:
            # Check for NaN
            nan_count = df[feature].isna().sum()
            if nan_count > 0:
                raise ValueError(f"Feature '{feature}' contains {nan_count} NaN values")
            
            # Check for infinite values
            inf_count = np.isinf(df[feature]).sum()
            if inf_count > 0:
                raise ValueError(f"Feature '{feature}' contains {inf_count} infinite values")
            
            # Check bounds
            min_val = df[feature].min()
            max_val = df[feature].max()
            
            if min_val < -1e-6 or max_val > 1 + 1e-6:
                raise ValueError(
                    f"Feature '{feature}' out of bounds: "
                    f"[{min_val:.6f}, {max_val:.6f}]"
                )
        
        logger.info("[OK] No NaN values in normalized features")
        logger.info("[OK] No infinite values in normalized features")
        logger.info("[OK] All normalized features within [0, 1] bounds")
        
        # Runtime assertions
        assert not df[numeric_features].isna().any().any(), "NaN values detected"
        assert not np.isinf(df[numeric_features].values).any(), "Infinite values detected"
        
        logger.info("[OK] All normalization assertions passed")
    
    except (ValueError, AssertionError) as e:
        logger.error(f"Normalization validation failed: {e}")
        print(f"\nERROR VALIDATION ERROR: {e}")
        sys.exit(1)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_normalization_summary(
    normalization_metadata: Dict[str, Dict[str, float]],
    report_path: Path,
    logger: logging.Logger
) -> None:
    """
    Generate normalization summary CSV report.
    
    Args:
        normalization_metadata: Metadata dictionary for all normalized features.
        report_path: Path to save report.
        logger: Logger instance.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    for feature, metadata in normalization_metadata.items():
        summary_data.append({
            "Feature": feature,
            "Original_Min": round(metadata["original_min"], 6),
            "Original_Max": round(metadata["original_max"], 6),
            "Normalized_Min": round(metadata["normalized_min"], 6),
            "Normalized_Max": round(metadata["normalized_max"], 6),
            "Normalized_Mean": round(metadata["normalized_mean"], 6),
            "Normalized_Std": round(metadata["normalized_std"], 6),
            "Method_Used": metadata["method"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Feature")
    summary_df.to_csv(report_path, index=False)
    
    logger.info(f"Normalization summary saved to {report_path}")


def generate_normalization_report(
    df_input: pd.DataFrame,
    df_output: pd.DataFrame,
    numeric_features: List[str],
    execution_time: float,
    report_path: Path,
    logger: logging.Logger
) -> None:
    """
    Generate comprehensive normalization report.
    
    Args:
        df_input: Original dataframe.
        df_output: Normalized dataframe.
        numeric_features: List of normalized features.
        execution_time: Total execution time in seconds.
        report_path: Path to save report.
        logger: Logger instance.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    reverse_count = sum(1 for f in numeric_features if f in REVERSE_NORMALIZED_FEATURES)
    standard_count = len(numeric_features) - reverse_count
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PMAF FEATURE NORMALIZATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET DIMENSIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Rows                 : {len(df_input)}\n")
        f.write(f"Total Columns              : {len(df_input.columns)}\n")
        f.write(f"Identifier Columns         : {len(IDENTIFIER_COLUMNS)}\n")
        f.write(f"Numeric Features Normalized: {len(numeric_features)}\n\n")
        
        f.write("NORMALIZATION METHODS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Standard Min-Max (x-min)/(max-min): {standard_count} features\n")
        f.write(f"Reverse Min-Max (max-x)/(max-min): {reverse_count} features\n\n")
        
        if reverse_count > 0:
            f.write("Reverse-normalized features:\n")
            for feat in REVERSE_NORMALIZED_FEATURES:
                if feat in numeric_features:
                    f.write(f"  • {feat}\n")
            f.write("\n")
        
        f.write("DATA INTEGRITY CHECKS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Missing identifiers (before): 0\n")
        f.write(f"Missing identifiers (after) : 0\n")
        f.write(f"Duplicate State-District pairs: 0\n")
        f.write(f"NaN values in normalized features: {df_output[numeric_features].isna().sum().sum()}\n")
        f.write(f"Infinite values in normalized features: {np.isinf(df_output[numeric_features].values).sum()}\n\n")
        
        f.write("NORMALIZATION BOUNDS CHECK\n")
        f.write("-" * 80 + "\n")
        bounds_violated = 0
        for feat in numeric_features:
            min_val = df_output[feat].min()
            max_val = df_output[feat].max()
            if min_val < -1e-6 or max_val > 1 + 1e-6:
                bounds_violated += 1
        
        f.write(f"Features within [0, 1] bounds: {len(numeric_features) - bounds_violated}/{len(numeric_features)}\n")
        f.write(f"Features with violations: {bounds_violated}\n\n")
        
        f.write("EXECUTION SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total execution time: {execution_time:.2f} seconds\n")
        f.write(f"Features normalized: {len(numeric_features)}\n")
        f.write(f"Rows processed: {len(df_output)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("VALIDATION STATUS: [OK] PASSED\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Normalization report saved to {report_path}")


# =============================================================================
# SAVE NORMALIZED DATASET
# =============================================================================

def save_dataset(
    df: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """
    Save normalized dataset to CSV.
    
    Args:
        df: Normalized dataframe.
        output_path: Path to save output.
        logger: Logger instance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving normalized dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    
    logger.info(f"Dataset saved successfully")
    logger.info(f"Output shape: {df.shape[0]} rows × {df.shape[1]} columns")


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_execution_summary(
    input_rows: int,
    numeric_feature_count: int,
    reverse_feature_count: int,
    execution_time: float
) -> None:
    """
    Print professional execution summary to console.
    
    Args:
        input_rows: Number of rows processed.
        numeric_feature_count: Total numeric features normalized.
        reverse_feature_count: Count of reverse-normalized features.
        execution_time: Total execution time in seconds.
    """
    print("\n")
    print("=" * 80)
    print("PMAF FEATURE NORMALIZATION")
    print("=" * 80)
    print()
    print(f"Input Dataset")
    print(f"  {input_rows} rows")
    print()
    print(f"↓")
    print()
    print(f"Numeric Features Normalized")
    print(f"  {numeric_feature_count} features")
    print(f"  {numeric_feature_count - reverse_feature_count} standard Min-Max")
    print(f"  {reverse_feature_count} reverse Min-Max")
    print()
    print(f"↓")
    print()
    print(f"Validation Passed")
    print(f"  [OK] No missing values")
    print(f"  [OK] No duplicates")
    print(f"  [OK] All features ∈ [0, 1]")
    print()
    print(f"↓")
    print()
    print(f"Reports Generated")
    print(f"  • normalization_summary.csv")
    print(f"  • normalization_report.txt")
    print()
    print(f"↓")
    print()
    print(f"Output Saved")
    print(f"  data/final/normalized_features.csv")
    print()
    print(f"Execution Time")
    print(f"  {execution_time:.2f} seconds")
    print()
    print("=" * 80)
    print("[OK] SUCCESS")
    print("=" * 80)
    print()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Execute complete normalization pipeline.
    """
    start_time = time.time()
    
    input_path = Path("data/final/pmaf_features.csv")
    output_path = Path("data/final/normalized_features.csv")
    log_path = Path("outputs/reports/normalization.log")
    summary_report_path = Path("outputs/tables/normalization_summary.csv")
    detailed_report_path = Path("outputs/reports/normalization_report.txt")
    
    logger = setup_logging(log_path)
    
    logger.info("=" * 80)
    logger.info("PMAF FEATURE NORMALIZATION PIPELINE STARTED")
    logger.info("=" * 80)
    
    try:
        # Load dataset
        df_input = load_dataset(input_path, logger)
        
        # Validate
        validate_dataset(df_input, logger)
        
        # Identify numeric features
        numeric_features = identify_numeric_features(df_input, logger)
        
        # Normalize
        logger.info("Normalizing features...")
        df_normalized, normalization_metadata = normalize_features(
            df_input,
            numeric_features,
            logger
        )
        
        # Validate normalization
        validate_normalization(df_normalized, numeric_features, logger)
        
        # Generate reports
        logger.info("Generating normalization summary...")
        generate_normalization_summary(
            normalization_metadata,
            summary_report_path,
            logger
        )
        
        logger.info("Generating detailed normalization report...")
        generate_normalization_report(
            df_input,
            df_normalized,
            numeric_features,
            time.time() - start_time,
            detailed_report_path,
            logger
        )
        
        # Save output
        save_dataset(df_normalized, output_path, logger)
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        reverse_count = sum(
            1 for f in numeric_features if f in REVERSE_NORMALIZED_FEATURES
        )
        print_execution_summary(
            len(df_input),
            len(numeric_features),
            reverse_count,
            elapsed_time
        )
        
        logger.info("=" * 80)
        logger.info("PMAF FEATURE NORMALIZATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\nERROR PIPELINE ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
