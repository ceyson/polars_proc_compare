"""Utility functions for batch processing of large datasets."""

import gc
import psutil
from typing import Dict, List
import polars as pl
from polars_proc_compare.comparison_engine import DataCompare
from polars_proc_compare.results import ComparisonResults


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dict with memory usage statistics (in MB):
        - used: Current memory used
        - available: Available memory
        - percent: Percentage of memory used
    """
    mem = psutil.virtual_memory()
    return {
        'used': mem.used / (1024 * 1024),
        'available': mem.available / (1024 * 1024),
        'percent': mem.percent
    }


def compare_in_batches(
    base_path: str,
    compare_path: str,
    key_columns: List[str],
    batch_size: int = 50,
    max_memory_percent: float = 80.0,
    chunk_size: int = 10_000,
    n_workers: int = 4,
    max_memory_usage: int = 2048,
    monitor_memory: bool = True,
    verbose: bool = True
) -> ComparisonResults:
    """Compare large datasets in column batches while maintaining report format.

    This function processes large datasets by comparing columns in batches to manage
    memory usage effectively. It maintains the same report format as regular comparisons.

    Args:
        base_path: Path to base parquet file
        compare_path: Path to comparison parquet file
        key_columns: List of columns to use as keys for matching rows
        batch_size: Number of columns to process in each batch
        max_memory_percent: Maximum memory usage percentage before forcing GC
        chunk_size: Number of rows to process at once
        n_workers: Number of parallel workers
        max_memory_usage: Maximum memory usage in MB for each batch
        monitor_memory: Whether to monitor memory usage
        verbose: Whether to print progress messages

    Returns:
        ComparisonResults object with full comparison results

    Raises:
        Exception: If there are errors during comparison
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    def check_memory() -> None:
        if monitor_memory:
            mem = get_memory_usage()
            if mem['percent'] > max_memory_percent:
                log(f"Memory usage high ({mem['percent']:.1f}%), running GC...")
                gc.collect()

    # Read schema and column names
    base_df = pl.read_parquet(base_path, n_rows=1)
    compare_df = pl.read_parquet(compare_path, n_rows=1)
    all_columns = [col for col in base_df.schema.keys() if col not in key_columns]

    # Initialize a ComparisonResults object with structure info
    base_nrows = pl.read_parquet(base_path, columns=[key_columns[0]]).height
    compare_nrows = pl.read_parquet(compare_path, columns=[key_columns[0]]).height
    
    # Calculate matched rows
    if key_columns:
        base_keys = set(pl.read_parquet(base_path, columns=key_columns).rows())
        compare_keys = set(pl.read_parquet(compare_path, columns=key_columns).rows())
        matched_rows = len(base_keys & compare_keys)
        base_only_rows = len(base_keys - compare_keys)
        compare_only_rows = len(compare_keys - base_keys)
    else:
        min_rows = min(base_nrows, compare_nrows)
        matched_rows = min_rows
        base_only_rows = max(0, base_nrows - min_rows)
        compare_only_rows = max(0, compare_nrows - min_rows)
    
    final_results = ComparisonResults()
    final_results.set_structure_results({
        'base_nrows': base_nrows,
        'compare_nrows': compare_nrows,
        'base_ncols': len(base_df.columns),
        'compare_ncols': len(compare_df.columns),
        'common_cols': list(set(base_df.columns) & set(compare_df.columns)),
        'base_only': list(set(base_df.columns) - set(compare_df.columns)),
        'compare_only': list(set(compare_df.columns) - set(base_df.columns)),
        'variable_types': {col: str(base_df.schema[col]) for col in base_df.columns},
        'matched_rows': matched_rows,
        'base_only_rows': base_only_rows,
        'compare_only_rows': compare_only_rows
    })

    # Initialize results storage
    all_differences = {}
    total_diffs = 0

    # Process columns in batches
    total_batches = (len(all_columns) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(all_columns), batch_size), 1):
        batch_columns = all_columns[i:i + batch_size]
        columns_to_read = key_columns + batch_columns

        log(f"Processing batch {batch_num}/{total_batches} "
            f"({len(batch_columns)} columns)")
        check_memory()

        try:
            # Read only needed columns
            base_df = pl.read_parquet(base_path, columns=columns_to_read)
            compare_df = pl.read_parquet(compare_path, columns=columns_to_read)
            
            dc = DataCompare(
                base_df=base_df,
                compare_df=compare_df,
                key_columns=key_columns,
                disk_mode=True,
                chunk_size=chunk_size,
                n_workers=n_workers,
                max_memory_usage=max_memory_usage,
                optimize_dtypes=True
            )

            # Run comparison for this batch
            results = dc.compare()

            # Accumulate results
            all_differences.update(results.comparison_results)
            total_diffs += results.total_differences

            log(f"Batch {batch_num} completed: "
                f"Found {results.total_differences} differences")
            check_memory()

        except Exception as e:
            log(f"Error in batch {batch_num}: {str(e)}")
            continue

    # Set the final results
    final_results.set_comparison_results(all_differences, total_diffs)
    return final_results
