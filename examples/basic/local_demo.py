"""Demo script for batch comparison with absolute paths."""

import polars as pl
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from polars_proc_compare.batch_utils import compare_in_batches, get_memory_usage

# Get absolute paths
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
data_dir = script_dir / "data/local_demo"
data_dir.mkdir(parents=True, exist_ok=True)

temp_dir = data_dir / "temp"
temp_dir.mkdir(exist_ok=True)

# Monitor initial memory
print("Initial memory state:")
mem = get_memory_usage()
print(f"Used: {mem['used']:.1f} MB")
print(f"Available: {mem['available']:.1f} MB")
print(f"Percent: {mem['percent']:.1f}%")

def create_sample_data(n_rows: int, n_cols: int, seed: int = 42) -> pl.DataFrame:
    """Create sample data with specified dimensions."""
    np.random.seed(seed)
    
    # Create base data
    data = {
        "id": pl.Series(range(n_rows)),
        "timestamp": pl.Series(
            [datetime(2024, 1, 1).timestamp() + i * 86400 for i in range(n_rows)]
        ).cast(pl.Datetime)
    }
    
    # Add columns with different types
    for i in range(n_cols):
        if i % 3 == 0:  # Integer columns
            data[f"int_col_{i}"] = pl.Series(
                np.random.randint(-10000, 10000, n_rows)
            )
        elif i % 3 == 1:  # Float columns
            data[f"float_col_{i}"] = pl.Series(
                np.random.normal(0, 100, n_rows)
            )
        else:  # String columns
            categories = [f"cat_{j}" for j in range(10)]
            data[f"cat_col_{i}"] = pl.Series(
                np.random.choice(categories, n_rows)
            )
    
    return pl.DataFrame(data)

# Create base dataset
print("Creating base dataset...")
base_df = create_sample_data(n_rows=10_000, n_cols=98)  # 98 + id + timestamp = 100 columns
base_path = data_dir / "base.parquet"
base_df.write_parquet(base_path)
print(f"Base dataset shape: {base_df.shape}")

# Create comparison with controlled differences
print("\nCreating comparison dataset...")
compare_df = base_df.clone()

# Modify some values
for col in base_df.columns[2:]:  # Skip id and timestamp
    if np.random.random() < 0.1:  # 10% of columns have differences
        rows_to_modify = np.random.choice(
            range(len(compare_df)), 
            size=int(len(compare_df) * 0.01),  # 1% of rows
            replace=False
        )
        if col.startswith('int'):
            compare_df = compare_df.with_columns([
                pl.when(pl.col('id').is_in(rows_to_modify))
                .then(pl.col(col) + np.random.randint(100, 1000))
                .otherwise(pl.col(col))
                .alias(col)
            ])
        elif col.startswith('float'):
            compare_df = compare_df.with_columns([
                pl.when(pl.col('id').is_in(rows_to_modify))
                .then(pl.col(col) * np.random.uniform(1.1, 2.0))
                .otherwise(pl.col(col))
                .alias(col)
            ])
        else:  # String columns
            compare_df = compare_df.with_columns([
                pl.when(pl.col('id').is_in(rows_to_modify))
                .then(pl.lit('modified'))
                .otherwise(pl.col(col))
                .alias(col)
            ])

compare_path = data_dir / "compare.parquet"
compare_df.write_parquet(compare_path)
print(f"Compare dataset shape: {compare_df.shape}")

# Show memory after data creation
print("\nMemory after data creation:")
mem = get_memory_usage()
print(f"Used: {mem['used']:.1f} MB")
print(f"Available: {mem['available']:.1f} MB")
print(f"Percent: {mem['percent']:.1f}%")

# Run batch comparison
print("Starting batch comparison...")
results = compare_in_batches(
    base_path=str(base_path.absolute()),
    compare_path=str(compare_path.absolute()),
    key_columns=["id"],
    batch_size=10,           # Process 10 columns at a time
    chunk_size=2_000,        # Process 2k rows at a time
    n_workers=1,             # Single worker to minimize memory
    max_memory_usage=512,    # 512MB memory limit per batch
    max_memory_percent=75.0, # Run GC at 75% memory usage
    monitor_memory=True,     # Enable memory monitoring
    verbose=True,           # Show progress messages
    temp_dir=temp_dir.absolute()  # Use absolute path for temp directory
)

# Save and display results
html_path = data_dir / "comparison.html"
results.to_html(html_path)
print(f"\nReport saved to: {html_path}")

print("\nComparison Summary:")
print(f"Total differences: {results.total_differences}")
print(f"Columns with differences: {len(results.comparison_results)}")
