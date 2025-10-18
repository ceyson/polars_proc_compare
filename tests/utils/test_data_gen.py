"""Test script to verify data generator."""

import numpy as np
import polars as pl
from polars_proc_compare.data_generator import create_delta_dataset

# Create a small test dataset
df = pl.DataFrame({
    "id": range(1, 101),
    "value": range(1, 101),
    "name": [f"name_{i}" for i in range(1, 101)]
})

# Create comparison with 5% differences
modified_df, stats = create_delta_dataset(
    df,
    delta_percentage=5.0,
    seed=42,
    exclude_columns=["id"]
)

print("Original DataFrame:")
print(df.head())
print("\nModified DataFrame:")
print(modified_df.head())
print("\nModification Stats:")
for k, v in stats.items():
    print(f"{k}: {v}")

# Find actual differences
print("\nActual Differences:")
for col in df.columns:
    if col != "id":
        # Get rows where values differ
        base_vals = df[col].to_numpy()
        mod_vals = modified_df[col].to_numpy()
        diff_indices = np.where(base_vals != mod_vals)[0]
        print(f"\n{col}: {len(diff_indices)} differences")
        if len(diff_indices) > 0:
            print("Sample differences (first 5):")
            for idx in diff_indices[:5]:
                print(f"  Row {idx}: {base_vals[idx]} -> {mod_vals[idx]}")
