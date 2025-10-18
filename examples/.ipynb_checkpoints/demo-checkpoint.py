import polars as pl
from polars_proc_compare import DataCompare
from polars_proc_compare.data_generator import create_delta_dataset

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
results.to_html("comparison_report.html")

# Export differences to CSV
results.to_csv("differences.csv")

print("\nComparison completed. Check comparison_report.html and differences.csv for results.")
