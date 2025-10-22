"""Data generator module for creating test datasets with controlled differences."""

import polars as pl
import numpy as np
from typing import Optional, Dict, Tuple, List


def create_delta_dataset(
    df: pl.DataFrame,
    delta_percentage: float,
    seed: Optional[int] = None,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pl.DataFrame, Dict]:
    """Create a modified copy of the input DataFrame with controlled deltas.

    Args:
        df: Input Polars DataFrame
        delta_percentage: Percentage of values to modify (0.0 to 100.0)
        seed: Random seed for reproducibility
        exclude_columns: List of column names to exclude from modifications

    Returns:
        Tuple containing:
        - Modified DataFrame
        - Dictionary with modification statistics
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None")
    if exclude_columns is None:
        exclude_columns = []
    if seed is not None:
        np.random.seed(seed)

    exclude_columns = exclude_columns or []
    modifiable_columns = [col for col in df.columns if col not in exclude_columns]

    # Initialize tracking
    modifications = {
        "total_rows": len(df),
        "total_columns": len(modifiable_columns),
        "total_cells": len(df) * len(modifiable_columns),
        "modified_cells": 0,
        "modified_columns": {},
        "modified_rows": set()
    }

    # Calculate modifications per column
    mods_per_column = int(len(df) * delta_percentage / 100)
    modified_series = {}

    for col_name in modifiable_columns:
        # Select random rows to modify
        row_indices = np.random.choice(len(df), size=mods_per_column, replace=False)
        modifications["modified_rows"].update(row_indices)
        modifications["modified_columns"][col_name] = mods_per_column
        modifications["modified_cells"] += mods_per_column

        # Get original values and handle nulls
        series = df[col_name]
        col_dtype = df.schema[col_name]

        # Apply modifications based on data type
        if col_dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
            # Convert to numpy array while preserving nulls
            values = series.to_numpy().copy()  # Create a writable copy
            mask = ~np.isnan(values)  # Track non-null values
            if col_dtype in [pl.Int64, pl.Int32]:
                # Only modify non-null values
                valid_indices = [idx for idx in row_indices if mask[idx]]
                if valid_indices:
                    # Create modified values array
                    mod_values = values.copy()
                    mod_values[valid_indices] += np.random.randint(-10, 11, size=len(valid_indices))
                    values = mod_values
            else:
                valid_indices = [idx for idx in row_indices if mask[idx]]
                if valid_indices:
                    # Create modified values array
                    mod_values = values.copy()
                    current_vals = values[valid_indices]
                    deltas = np.random.uniform(-1, 1, size=len(valid_indices))
                    mod_values[valid_indices] += deltas * np.where(current_vals != 0, np.abs(current_vals), 1)
                    values = mod_values
        elif col_dtype == pl.Boolean:
            # Create a copy for boolean values
            mod_values = values.copy()
            mod_values[row_indices] = ~mod_values[row_indices]
            values = mod_values
        else:
            # Handle string modifications while preserving type
            mod_values = values.copy()
            mod_values[row_indices] = np.array([f"{v}_modified" for v in values[row_indices]], dtype=str)
            values = mod_values

        # Create new series
        modified_series[col_name] = pl.Series(name=col_name, values=values)

    # Create modified DataFrame efficiently
    modified_df = df.clone()
    if modified_series:
        modified_df = modified_df.with_columns([modified_series[col] for col in modified_series])

    # Convert modified_rows to count
    modifications["modified_rows"] = len(modifications["modified_rows"])

    return modified_df, modifications
