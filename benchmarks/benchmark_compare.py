"""Benchmark module for comparing performance with different dataset sizes."""

import json
from pathlib import Path
from time import time
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from polars_proc_compare import DataCompare
from polars_proc_compare.data_generator import create_delta_dataset


def generate_test_dataset(
        n_rows: int,
        n_numeric_cols: int = 10,
        n_string_cols: int = 5,
        seed: int = 42,
) -> pl.DataFrame:
    """Generate test dataset with specified size and column types.

    Args:
        n_rows: Number of rows in the dataset
        n_numeric_cols: Number of numeric columns to generate
        n_string_cols: Number of string columns to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with generated test data
    """
    np.random.seed(seed)

    # Generate columns one by one to avoid memory issues
    columns = [pl.Series("id", range(1, n_rows + 1))]

    # Add numeric columns
    for i in range(n_numeric_cols):
        columns.append(
            pl.Series(f"num_{i}", np.random.normal(0, 1, n_rows))
        )

    # Add string columns
    for i in range(n_string_cols):
        columns.append(
            pl.Series(
                f"str_{i}",
                [f"val_{np.random.randint(1000)}" for _ in range(n_rows)]
            )
        )

    return pl.DataFrame(columns)


def benchmark_single_configuration(
        base_df: pl.DataFrame,
        compare_df: pl.DataFrame,
        chunk_size: Optional[int] = None,
        n_runs: int = 3,
) -> Dict[str, float]:
    """Run benchmark for a single configuration.

    Args:
        base_df: Base DataFrame to compare
        compare_df: DataFrame to compare against base
        chunk_size: Size of chunks for processing
        n_runs: Number of benchmark runs

    Returns:
        Dictionary containing benchmark statistics
    """
    times = []
    memory_usage = []

    for _ in range(n_runs):
        start_time = time()
        dc = DataCompare(
            base_df=base_df,
            compare_df=compare_df,
            key_columns=["id"],
            chunk_size=chunk_size,
        )
        results_obj = dc.compare()

        end_time = time()
        times.append(end_time - start_time)

        # Rough memory estimate based on number of cells
        memory_usage.append(
            len(base_df) * len(base_df.columns) * 8 +  # Assuming 8 bytes per value
            len(compare_df) * len(compare_df.columns) * 8
        )

    stats = {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "avg_memory_mb": sum(memory_usage) / len(memory_usage) / (1024 * 1024),
        "total_differences": results_obj.total_differences,
        "throughput": len(base_df) / (sum(times) / len(times)),
    }

    return stats


def run_benchmark(
        sizes: List[int],
        delta_percentages: List[float],
        chunk_sizes: List[int],
        n_runs: int = 3,
) -> Dict[int, Dict[float, Dict[int, Dict[str, float]]]]:  # Nested dict structure
    """Run benchmarks with different configurations.

    Args:
        sizes: List of dataset sizes to test
        delta_percentages: List of difference percentages to test
        chunk_sizes: List of chunk sizes to test
        n_runs: Number of runs per configuration

    Returns:
        Nested dictionary containing benchmark results
    """
    results = {}

    for size in sizes:
        print(f"\nProcessing dataset size: {size:,} rows")
        results[size] = {}
        base_df = generate_test_dataset(size)

        for delta_pct in delta_percentages:
            print(f"  Testing {delta_pct}% differences")
            results[size][delta_pct] = {}
            # Create comparison dataset with differences
            compare_df, _ = create_delta_dataset(
                df=base_df,
                delta_percentage=delta_pct,
                seed=42,
                exclude_columns=["id"],
            )

            for chunk_size in chunk_sizes:
                print(f"    Using chunk size: {chunk_size:,}")
                stats = benchmark_single_configuration(
                    base_df=base_df,
                    compare_df=compare_df,
                    chunk_size=chunk_size,
                    n_runs=n_runs,
                )
                results[size][delta_pct][chunk_size] = stats

    return results


def _print_benchmark_summary(
        results: Dict[int, Dict[float, Dict[int, Dict[str, float]]]],
        sizes: List[int],
        delta_percentages: List[float],
        chunk_sizes: List[int],
) -> None:
    """Print a formatted summary of benchmark results.

    Args:
        results: Nested dictionary of benchmark results
        sizes: List of dataset sizes tested
        delta_percentages: List of difference percentages tested
        chunk_sizes: List of chunk sizes tested
    """
    for size in sizes:
        print(f"\nDataset Size: {size:,} rows")
        print("-" * 40)
        for delta_pct in delta_percentages:
            print(f"\nDelta Percentage: {delta_pct}%")
            for chunk_size in chunk_sizes:
                r = results[size][delta_pct][chunk_size]
                print(f"  Chunk Size {chunk_size:,}:")
                print(
                    f"    Average Time: {r['avg_time']:.2f}s "
                    f"(min: {r['min_time']:.2f}s, max: {r['max_time']:.2f}s)"
                )
                print(f"    Memory Usage: {r['avg_memory_mb']:.1f}MB")
                print(
                    f"    Differences Found: {r['total_differences']:,} "
                    f"({(r['total_differences'] / size * 100):.1f}% of rows)"
                )
                print(f"    Throughput: {r['throughput']:,.0f} rows/second")
                print(
                    f"    Memory Efficiency: "
                    f"{(r['throughput'] / r['avg_memory_mb']):,.0f} rows/second/MB"
                )


def main() -> None:
    """Run the benchmark suite and save results."""
    # Test configurations
    sizes = [500_000, 1_000_000, 2_000_000]  # Even larger datasets
    delta_percentages = [1.0, 5.0]  # Fewer percentages for faster testing
    chunk_sizes = [50_000, 100_000]  # Focus on larger chunks
    n_runs = 3  # Number of runs for each configuration

    print("Running benchmarks...")
    print("=" * 80)
    results = run_benchmark(
        sizes=sizes,
        delta_percentages=delta_percentages,
        chunk_sizes=chunk_sizes,
        n_runs=n_runs,
    )

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nBenchmark Results Summary:")
    print("=" * 80)
    _print_benchmark_summary(results, sizes, delta_percentages, chunk_sizes)


if __name__ == "__main__":
    main()
