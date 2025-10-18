import polars as pl
import pytest
from polars_proc_compare import DataCompare
from polars_proc_compare.data_generator import create_delta_dataset

@pytest.fixture
def sample_dataframes():
    # Create a sample DataFrame
    base_df = pl.DataFrame({
        "id": range(1, 101),
        "name": [f"name_{i}" for i in range(1, 101)],
        "value": [float(i) for i in range(1, 101)],
        "category": ["A" if i % 2 == 0 else "B" for i in range(1, 101)]
    })
    
    # Create modified DataFrame with known deltas
    compare_df, modifications = create_delta_dataset(
        base_df,
        delta_percentage=5.0,
        seed=42,
        exclude_columns=["id"]
    )
    
    return base_df, compare_df, modifications

def test_structure_comparison(sample_dataframes):
    base_df, compare_df, _ = sample_dataframes
    
    # Test with identical structure
    dc = DataCompare(base_df, compare_df)
    results = dc.compare()
    
    assert len(results.structure_results["base_only"]) == 0
    assert len(results.structure_results["compare_only"]) == 0
    assert len(results.structure_results["common_cols"]) == 4
    
    # Test with different structure
    modified_compare = compare_df.drop("category")
    dc = DataCompare(base_df, modified_compare)
    results = dc.compare()
    
    assert "category" in results.structure_results["base_only"]
    assert len(results.structure_results["compare_only"]) == 0

def test_value_comparison(sample_dataframes):
    base_df, compare_df, modifications = sample_dataframes
    
    dc = DataCompare(base_df, compare_df, key_columns=["id"])
    results = dc.compare()
    
    # Verify number of differences matches modifications
    total_differences = sum(stat["n_differences"] for stat in results.statistics.values())
    assert total_differences == modifications["modified_cells"]

def test_html_report_generation(sample_dataframes, tmp_path):
    base_df, compare_df, _ = sample_dataframes
    
    dc = DataCompare(base_df, compare_df)
    results = dc.compare()
    
    report_path = tmp_path / "report.html"
    results.to_html(str(report_path))
    
    assert report_path.exists()
    assert report_path.stat().st_size > 0

def test_csv_export(sample_dataframes, tmp_path):
    base_df, compare_df, _ = sample_dataframes
    
    dc = DataCompare(base_df, compare_df)
    results = dc.compare()
    
    csv_path = tmp_path / "differences.csv"
    results.to_csv(str(csv_path))
    
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0
