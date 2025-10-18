"""Benchmark module for comparing performance with different dataset sizes."""

import json
from pathlib import Path
from time import time
from typing import Dict, List

import numpy as np
import polars as pl

from polars_proc_compare import DataCompare
from polars_proc_compare.data_generator import create_delta_dataset

def generate_test_dataset(
    n_rows: int,
    n_numeric_cols: int = 5,
    n_string_cols: int = 3,
    seed: int = 42
) -> pl.DataFrame:
    """Generate test dataset with specified size and column types."""
    np.random.seed(seed)
    
    data = {
        "id": range(1, n_rows + 1),
        **{
            f"num_{i}": np.random.normal(0, 1, n_rows)
            for i in range(n_numeric_cols)
        },
        **{
            f"str_{i}": [f"val_{np.random.randint(1000)}" for _ in range(n_rows)]
            for i in range(n_string_cols)
        }
    }
    
    return pl.DataFrame(data)

def run_benchmark(
    sizes: List[int],
    delta_percentages: List[float],
    chunk_sizes: List[int],
    n_runs: int = 3
) -> Dict:
    """Run benchmarks with different configurations."""
    results = {}
    
    for size in sizes:
        results[size] = {}
        base_df = generate_test_dataset(size)
        
        for delta_pct in delta_percentages:
            results[size][delta_pct] = {}
            compare_df, _ = create_delta_dataset(
                base_df,
                delta_percentage=delta_pct,
                seed=42,
                exclude_columns=["id"]
            )
            
            for chunk_size in chunk_sizes:
                times = []
                memory_usage = []
                
                for _ in range(n_runs):
                    start_time = time()
                    
                    dc = DataCompare(
                        base_df,
                        compare_df,
                        key_columns=["id"],
                        chunk_size=chunk_size
                    )
                    results_obj = dc.compare()
                    
                    end_time = time()
                    times.append(end_time - start_time)
                    
                    # Track memory usage using DataFrame size estimates
                    memory_usage.append(
                        base_df.estimated_size() +
                        compare_df.estimated_size()
                    )
                
                results[size][delta_pct][chunk_size] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage) / (1024 * 1024),
                    "total_differences": results_obj.total_differences
                }
    
    return results

def main():
    # Test configurations
    sizes = [10_000, 100_000, 1_000_000]
    delta_percentages = [1.0, 5.0, 10.0]
    chunk_sizes = [10_000, 50_000, 100_000]
    
    print("Running benchmarks...")
    results = run_benchmark(sizes, delta_percentages, chunk_sizes)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("=" * 80)
    for size in sizes:
        print(f"\nDataset Size: {size:,} rows")
        print("-" * 40)
        for delta_pct in delta_percentages:
            print(f"\nDelta Percentage: {delta_pct}%")
            for chunk_size in chunk_sizes:
                r = results[size][delta_pct][chunk_size]
                print(f"  Chunk Size {chunk_size:,}:")
                print(f"    Average Time: {r['avg_time']:.2f}s")
                print(f"    Memory Usage: {r['avg_memory_mb']:.1f}MB")
                print(f"    Differences Found: {r['total_differences']:,}")


if __name__ == "__main__":
    main()
