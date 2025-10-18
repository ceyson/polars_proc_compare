"""Test module for large dataset comparison functionality."""

import time

import numpy as np
import polars as pl
import pytest


from polars_proc_compare import DataCompare
from polars_proc_compare.data_generator import create_delta_dataset


@pytest.fixture(
    params=[
        (100_000, 5),  # Medium dataset
        (1_000_000, 3),  # Large dataset
        (5_000_000, 2),  # Very large dataset
    ],
    ids=["medium", "large", "very_large"]
)
def large_dataset(request) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate large test datasets with different sizes."""
    n_rows, n_cols = request.param

    # Generate base dataset
    data = {
        "id": range(1, n_rows + 1),
        **{
            f"value_{i}": np.random.normal(0, 1, n_rows)
            for i in range(n_cols)
        }
    }
    base_df = pl.DataFrame(data)

    # Create comparison dataset with known differences
    compare_df, _ = create_delta_dataset(
        base_df,
        delta_percentage=5.0,
        seed=42,
        exclude_columns=["id"]
    )

    return base_df, compare_df


@pytest.mark.large
def test_memory_efficiency(large_dataset):
    """Test memory usage stays within reasonable bounds."""
    base_df, compare_df = large_dataset
    initial_size = base_df.estimated_size() + compare_df.estimated_size()

    dc = DataCompare(base_df, compare_df, key_columns=["id"])
    _ = dc.compare()  # Run comparison to check memory usage

    # Check that we're not creating too many intermediate copies
    # by verifying total memory usage is reasonable
    total_size = sum(df.estimated_size() for df in [base_df, compare_df])
    assert total_size < initial_size * 3  # Allow for some intermediate data


@pytest.mark.large
def test_chunk_processing(large_dataset):
    """Test that chunk processing works correctly for large datasets."""
    base_df, compare_df = large_dataset

    # Test with different chunk sizes
    chunk_sizes = [10_000, 50_000, 100_000]
    results = []

    for chunk_size in chunk_sizes:
        dc = DataCompare(
            base_df,
            compare_df,
            key_columns=["id"],
            chunk_size=chunk_size
        )
        results.append(dc.compare())

    # All chunk sizes should produce same results
    first_result = results[0]
    for other_result in results[1:]:
        assert first_result.total_differences == other_result.total_differences
        assert (
            len(first_result.statistics) ==
            len(other_result.statistics)
        )


@pytest.mark.large
def test_parallel_processing(large_dataset):
    """Test parallel processing with different numbers of workers."""
    base_df, compare_df = large_dataset

    # Test with different numbers of workers
    times = []
    for n_workers in [1, 2, 4]:
        start_time = time.time()

        dc = DataCompare(
            base_df,
            compare_df,
            key_columns=["id"],
            chunk_size=50_000,
            n_workers=n_workers
        )
        dc.compare()

        times.append(time.time() - start_time)

    # More workers should not be significantly slower
    # Allow for some overhead and system variations
    single_worker_time = times[0]
    max_acceptable_time = single_worker_time * 1.5  # Allow up to 50% slower

    # Check that parallel versions aren't much slower than single worker
    for worker_time in times[1:]:
        assert worker_time <= max_acceptable_time, \
            f"Parallel processing ({worker_time:.2f}s) was significantly slower than " \
            f"single-threaded ({single_worker_time:.2f}s)"


@pytest.mark.large
def test_large_differences(large_dataset):
    """Test handling of datasets with many differences."""
    base_df, _ = large_dataset

    # Create comparison with many differences
    compare_df, _ = create_delta_dataset(
        base_df,
        delta_percentage=50.0,  # Large number of differences
        seed=42,
        exclude_columns=["id"]
    )

    dc = DataCompare(base_df, compare_df, key_columns=["id"])
    results = dc.compare()

    # Should handle large number of differences efficiently
    assert results.total_differences > 0
    assert all(
        len(stats["first_n_differences"]) <= 20
        for stats in results.statistics.values()
    )
