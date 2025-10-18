import polars as pl
import numpy as np
from typing import Optional, Dict, Tuple, List

def create_delta_dataset(
    df: pl.DataFrame,
    delta_percentage: float,
    seed: Optional[int] = None,
    exclude_columns: Optional[List[str]] = None,
    batch_size: int = 100_000
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
    if seed is not None:
        np.random.seed(seed)
    
    exclude_columns = exclude_columns or []
    modifiable_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Create modification masks for each column type
    data = {}
    modifications = {
        "total_rows": len(df),
        "total_columns": len(modifiable_columns),
        "total_cells": len(df) * len(modifiable_columns),
        "modified_cells": 0,
        "modified_columns": {},
        "modified_rows": set()
    }
    
    # Calculate number of modifications per column
    mods_per_column = int(len(df) * delta_percentage / 100)
    
    for col_name in modifiable_columns:
        # Generate random row indices for this column
        row_indices = np.random.choice(len(df), size=mods_per_column, replace=False)
        modifications["modified_rows"].update(row_indices)
        modifications["modified_columns"][col_name] = mods_per_column
        modifications["modified_cells"] += mods_per_column
        
        # Get current values
        current_values = df[col_name]
        new_values = current_values.clone()
        
        # Generate modifications based on data type
        col_dtype = df.schema[col_name]
        if col_dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
            if col_dtype in [pl.Int64, pl.Int32]:
                deltas = np.random.randint(-10, 11, size=mods_per_column)
                new_values = new_values.to_numpy()
                new_values[row_indices] += deltas
            else:
                current_vals = current_values[row_indices].to_numpy()
                deltas = np.random.uniform(-1, 1, size=mods_per_column) * \
                         np.where(current_vals != 0, np.abs(current_vals), 1)
                new_values = new_values.to_numpy()
                new_values[row_indices] += deltas
        elif col_dtype == pl.Boolean:
            values = new_values.to_numpy().copy()  # Create a writeable copy
            values[row_indices] = ~values[row_indices]
            new_values = values
        else:
            values = new_values.to_numpy().copy()  # Create a writeable copy
            values[row_indices] = [str(v) + "_modified" for v in values[row_indices]]
            new_values = values
        
        data[col_name] = pl.Series(name=col_name, values=new_values)
    
    # Create modified DataFrame efficiently
    modified_df = df.clone()
    if data:
        modified_df = modified_df.with_columns([data[col] for col in data.keys()])
    
    # Convert modified_rows to count
    modifications["modified_rows"] = len(modifications["modified_rows"])
    
    return modified_df, modifications
