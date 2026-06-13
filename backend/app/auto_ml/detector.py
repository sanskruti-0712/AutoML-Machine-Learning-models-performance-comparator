import pandas as pd
import numpy as np

def detect_problem_type(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Detects the ML problem type (Classification, Regression, or Time Series)
    and identifies datetime columns.
    """
    # 1. Detect datetime columns
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == 'object':
            # Try parsing a sample
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                try:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    # If at least 80% of non-null values parse as dates, treat as datetime
                    if parsed.notna().sum() / len(sample) >= 0.8:
                        datetime_cols.append(col)
                except Exception:
                    pass

    # If target column is not provided, pick the last column that isn't datetime
    if not target_col or target_col not in df.columns:
        remaining_cols = [c for c in df.columns if c not in datetime_cols]
        target_col = remaining_cols[-1] if remaining_cols else df.columns[-1]

    # Analyze target column
    target_series = df[target_col].dropna()
    unique_vals = target_series.nunique()
    total_vals = len(target_series)
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    # Heuristics for problem detection
    if unique_vals <= 2:
        problem_type = "classification"
        subtype = "binary"
    elif not is_numeric:
        problem_type = "classification"
        subtype = "multiclass"
    else:
        # Numeric target: check cardinality
        # If integer with low cardinality (e.g. <= 10), treat as classification
        if pd.api.types.is_integer_dtype(target_series) and unique_vals <= 10:
            problem_type = "classification"
            subtype = "multiclass"
        else:
            problem_type = "regression"
            subtype = "continuous"

    # Suggest if it could be a Time Series problem
    has_time_series = len(datetime_cols) > 0
    
    return {
        "detected_target": target_col,
        "problem_type": problem_type,
        "subtype": subtype,
        "datetime_columns": datetime_cols,
        "has_time_series": has_time_series,
        "unique_target_values": unique_vals,
        "is_numeric_target": bool(is_numeric),
        "total_rows": len(df),
        "total_cols": len(df.columns)
    }
