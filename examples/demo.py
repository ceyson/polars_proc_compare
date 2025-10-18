import polars as pl
from pathlib import Path
from polars_proc_compare import DataCompare
from polars_proc_compare.data_generator import create_delta_dataset

# Create results directory if it doesn't exist
results_dir = Path("data")
results_dir.mkdir(exist_ok=True)

# Create base dataset
base_df = pl.DataFrame({
    "id": range(1, 1001),
    "name": [f"name_{i}" for i in range(1, 1001)],
    "value": [float(i) for i in range(1, 1001)],
    "category": ["A" if i % 2 == 0 else "B" for i in range(1, 1001)]
})

# Create comparison dataset with 5% differences
compare_df, modifications = create_delta_dataset(
    base_df,
    delta_percentage=5.0,
    seed=42,
    exclude_columns=["id"]
)

print("Modification Statistics:")
for key, value in modifications.items():
    print(f"{key}: {value}")

# Create comparison object
dc = DataCompare(base_df, compare_df, key_columns=["id"])

# Run comparison
results = dc.compare()

# Generate HTML report
results.to_html(results_dir / "comparison_report.html")

# Export differences to CSV
results.to_csv(results_dir / "differences.csv")

print("\nComparison completed. Check data/comparison_report.html and data/differences.csv for results.")
