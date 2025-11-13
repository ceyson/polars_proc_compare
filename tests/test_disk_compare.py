"""Tests for disk-based comparison functionality."""

import polars as pl
import pytest
from pathlib import Path
import tempfile
import shutil
from polars_proc_compare import DataCompare
import numpy as np

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def large_dataframes():
    """Create large test dataframes."""
    # Create dataframes with 1M rows
    size = 1_000_000
    np.random.seed(42)
    
    df1 = pl.DataFrame({
        'id': range(size),
        'value1': np.random.randint(-1000, 1000, size),
        'value2': np.random.random(size),
        'category': np.random.choice(['A', 'B', 'C'], size)
    })
    
    # Create similar df with some differences
    df2 = pl.DataFrame({
        'id': range(size),
        'value1': np.random.randint(-1000, 1000, size),  # Different values
        'value2': df1['value2'] + np.random.normal(0, 0.1, size),  # Slightly modified
        'category': df1['category']  # Same categories
    })
    
    return df1, df2

def test_disk_mode_basic(temp_dir, large_dataframes):
    """Test basic disk mode functionality."""
    df1, df2 = large_dataframes
    
    # Save to parquet
    df1_path = Path(temp_dir) / "df1.parquet"
    df2_path = Path(temp_dir) / "df2.parquet"
    df1.write_parquet(df1_path)
    df2.write_parquet(df2_path)
    
    # Compare using disk mode
    compare = DataCompare(
        base_df=str(df1_path),
        compare_df=str(df2_path),
        disk_mode=True,
        key_columns=['id'],
        temp_dir=temp_dir
    )
    
    results = compare.compare()
    
    # Verify structure results
    assert results.structure_results["base_nrows"] == len(df1)
    assert results.structure_results["compare_nrows"] == len(df2)
    assert set(results.structure_results["common_cols"]) == {'id', 'value1', 'value2', 'category'}
    
    # Verify some differences were found
    assert results.total_differences > 0
    assert 'value1' in results.comparison_results
    assert 'value2' in results.comparison_results
    assert 'category' not in results.comparison_results  # Should be identical

def test_memory_optimization(temp_dir):
    """Test memory optimization features."""
    # Create dataframe with different types
    df1 = pl.DataFrame({
        'small_int': np.random.randint(-100, 100, 10000),  # Should be Int8
        'med_int': np.random.randint(-1000, 1000, 10000),  # Should be Int16
        'big_float': np.random.random(10000),  # Should stay Float64
        'small_float': np.random.random(10000) * 0.001,  # Could be Float32
        'category': np.random.choice(['A', 'B', 'C'], 10000)  # Should be categorical
    })
    
    df2 = df1.clone()
    df2['small_int'] = df2['small_int'] + 1  # Create some differences
    
    # Save to parquet
    df1_path = Path(temp_dir) / "df1.parquet"
    df2_path = Path(temp_dir) / "df2.parquet"
    df1.write_parquet(df1_path)
    df2.write_parquet(df2_path)
    
    # Compare with optimization
    compare = DataCompare(
        base_df=str(df1_path),
        compare_df=str(df2_path),
        disk_mode=True,
        optimize_dtypes=True,
        temp_dir=temp_dir
    )
    
    results = compare.compare()
    
    # Verify differences were found
    assert results.total_differences > 0
    assert 'small_int' in results.comparison_results

def test_chunked_processing(temp_dir, large_dataframes):
    """Test chunked processing with different chunk sizes."""
    df1, df2 = large_dataframes
    
    # Save to parquet
    df1_path = Path(temp_dir) / "df1.parquet"
    df2_path = Path(temp_dir) / "df2.parquet"
    df1.write_parquet(df1_path)
    df2.write_parquet(df2_path)
    
    # Compare with small chunks
    compare_small = DataCompare(
        base_df=str(df1_path),
        compare_df=str(df2_path),
        disk_mode=True,
        chunk_size=10000,
        temp_dir=temp_dir
    )
    
    # Compare with large chunks
    compare_large = DataCompare(
        base_df=str(df1_path),
        compare_df=str(df2_path),
        disk_mode=True,
        chunk_size=100000,
        temp_dir=temp_dir
    )
    
    results_small = compare_small.compare()
    results_large = compare_large.compare()
    
    # Results should be the same regardless of chunk size
    assert results_small.total_differences == results_large.total_differences
    for col in results_small.comparison_results:
        assert results_small.comparison_results[col]["n_differences"] == \
               results_large.comparison_results[col]["n_differences"]

def test_key_based_comparison(temp_dir):
    """Test key-based comparison with unordered data."""
    # Create dataframes with shuffled order
    size = 10000
    np.random.seed(42)
    
    df1 = pl.DataFrame({
        'id': np.random.permutation(size),
        'value': np.random.random(size)
    })
    
    # Create df2 with same IDs but different order
    df2 = pl.DataFrame({
        'id': np.random.permutation(df1['id'].to_numpy()),
        'value': np.random.random(size)
    })
    
    # Save to parquet
    df1_path = Path(temp_dir) / "df1.parquet"
    df2_path = Path(temp_dir) / "df2.parquet"
    df1.write_parquet(df1_path)
    df2.write_parquet(df2_path)
    
    # Compare using key-based comparison
    compare = DataCompare(
        base_df=str(df1_path),
        compare_df=str(df2_path),
        disk_mode=True,
        key_columns=['id'],
        temp_dir=temp_dir
    )
    
    results = compare.compare()
    
    # Verify differences were found despite different order
    assert results.total_differences > 0
    assert 'value' in results.comparison_results
    
    # Verify all rows were matched by key
    assert results.structure_results["matched_rows"] == size
    assert results.structure_results["base_only_rows"] == 0
    assert results.structure_results["compare_only_rows"] == 0
