"""Comparison engine for Polars DataFrames."""

import polars as pl
from typing import Optional, Dict, List, Tuple
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from .results import ComparisonResults


class DataCompare:
    def __init__(
        self,
        base_df: pl.DataFrame,
        compare_df: pl.DataFrame,
        key_columns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        n_workers: Optional[int] = None,
        cache_size: int = 1024,
        use_streaming: bool = True,
        memory_limit: Optional[int] = None  # in MB
    ):
        """Initialize the comparison engine."""
        self.base_df = base_df
        self.compare_df = compare_df
        self.key_columns = key_columns
        self.n_workers = n_workers or min(32, mp.cpu_count() * 2)  # More aggressive parallelization
        self.cache_size = cache_size
        self.use_streaming = use_streaming
        self.memory_limit = memory_limit
        self.results = ComparisonResults()

        # Calculate optimal chunk size
        total_rows = max(len(base_df), len(compare_df))
        if chunk_size is None:
            # Dynamic chunk sizing based on dataset size and available CPUs
            if total_rows < 100_000:
                self.chunk_size = 10_000
            elif total_rows < 1_000_000:
                self.chunk_size = 50_000
            else:
                self.chunk_size = 100_000

            # Adjust for number of workers
            self.chunk_size = max(self.chunk_size, total_rows // (self.n_workers * 4))
        else:
            self.chunk_size = chunk_size

        # Adjust chunk size based on memory limit if specified
        if memory_limit:
            estimated_row_size = (
                sum(self.base_df.estimated_size() / len(self.base_df))
                for df in [self.base_df, self.compare_df]
            )
            max_rows = (memory_limit * 1024 * 1024) / estimated_row_size
            self.chunk_size = min(self.chunk_size, int(max_rows / 3))

    def _normalize_column(self, df: pl.DataFrame, col: str) -> pl.Series:
        """Normalize a column to a consistent type."""
        return df[col]  # Return column as-is

    def _compare_structure(self) -> Dict:
        """Compare the structure of both DataFrames."""
        base_cols = set(self.base_df.columns)
        comp_cols = set(self.compare_df.columns)

        return {
            "common_cols": list(base_cols & comp_cols),
            "base_only": list(base_cols - comp_cols),
            "compare_only": list(comp_cols - base_cols),
            "base_schema": self.base_df.schema,
            "compare_schema": self.compare_df.schema,
            "base_nrows": len(self.base_df),
            "compare_nrows": len(self.compare_df),
            "base_ncols": len(base_cols),
            "compare_ncols": len(comp_cols),
            "variable_types": {col: str(self.base_df.schema[col]) for col in base_cols & comp_cols},
            "matched_rows": 0,  # Will be updated during value comparison
            "base_only_rows": 0,  # Will be updated during value comparison
            "compare_only_rows": 0  # Will be updated during value comparison
        }

    def _calculate_column_stats(self, diff_rows: pl.DataFrame, base_col: str, comp_col: str) -> Dict:
        """Calculate statistics for a column with differences."""
        # Use lazy evaluation for better performance
        diff_lazy = diff_rows.lazy()

        col_stats = {
            "n_differences": len(diff_rows),
            "first_n_differences": []
        }

        # Get all differences
        sample_diff = diff_lazy.collect()

        if diff_rows[base_col].dtype.is_numeric():
            # Numeric comparisons using normalized types
            diffs = (
                sample_diff
                .with_columns([
                    ((pl.col(comp_col) - pl.col(base_col))
                     .round(4)
                     .alias("abs_diff")),
                    (pl.when(pl.col(base_col) != 0)
                     .then(((pl.col(comp_col) - pl.col(base_col)) / pl.col(base_col) * 100).round(2))
                     .otherwise(None)
                     .alias("pct_diff"))
                ])
            )

            # Include row numbers
            select_cols = [
                pl.col("__row_id").alias("obs"),  # Use actual row number
                pl.col(base_col).alias("base"),
                pl.col(comp_col).alias("compare"),
                pl.col("abs_diff"),
                pl.col("pct_diff")
            ]
            col_stats["first_n_differences"] = diffs.select(select_cols).rows(named=True)

            # Calculate overall statistics for non-null values only
            valid_diffs = (
                diff_rows
                .filter(pl.col(base_col).is_not_null() & pl.col(comp_col).is_not_null())
                .select([(pl.col(comp_col) - pl.col(base_col)).abs().alias("diff")])
            )
            if len(valid_diffs) > 0:
                col_stats.update({
                    "max_diff": round(float(valid_diffs["diff"].max()), 4),
                    "mean_diff": round(float(valid_diffs["diff"].mean()), 4)
                })
            else:
                col_stats.update({
                    "max_diff": None,
                    "mean_diff": None
                })
        else:
            # Non-numeric comparisons with actual row numbers
            select_cols = [
                pl.col("__row_id").alias("obs"),  # Use actual row number
                pl.col(base_col).alias("base"),
                pl.col(comp_col).alias("compare")
            ]
            col_stats["first_n_differences"] = sample_diff.select(select_cols).rows(named=True)

        return col_stats

    def _process_chunk_batch(self, chunks: List[Tuple[pl.LazyFrame, pl.LazyFrame]]) -> List[Dict]:
        """Process a batch of chunks in parallel."""
        results = []
        for base_chunk, compare_chunk in chunks:
            chunk_result = self._process_chunk(base_chunk, compare_chunk)
            results.append(chunk_result)
        return results

    def _process_chunk(self, base_chunk: pl.DataFrame | pl.LazyFrame, compare_chunk: pl.DataFrame | pl.LazyFrame) -> Dict:
        """Process a single chunk of data using streaming."""
        # Convert to DataFrame if needed
        base_data = base_chunk.collect() if isinstance(base_chunk, pl.LazyFrame) else base_chunk
        compare_data = compare_chunk.collect() if isinstance(compare_chunk, pl.LazyFrame) else compare_chunk

        # Normalize columns
        base_df = pl.DataFrame({
            col: self._normalize_column(base_data, col)
            for col in base_data.columns
        })

        compare_df = pl.DataFrame({
            col: self._normalize_column(compare_data, col)
            for col in compare_data.columns
        })

        # Always add row index and use it in merged results
        base_df = base_df.with_row_count("__row_id")
        compare_df = compare_df.with_row_count("__row_id")

        # Join on key columns if provided, otherwise use row index
        if self.key_columns:
            # For key-based comparison, keep row numbers from base
            merged = base_df.join(
                compare_df.drop("__row_id"),  # Drop compare's row id to avoid conflict
                on=self.key_columns,
                how="outer",
                suffix="_compare"
            )
        else:
            # For position-based comparison, use row index
            merged = base_df.join(
                compare_df,
                on="__row_id",
                how="outer",
                suffix="_compare"
            )

        chunk_stats = {}
        for col in self.results.structure_results["common_cols"]:
            if col not in (self.key_columns or []):
                base_col = col
                comp_col = f"{col}_compare"

                if comp_col in merged.columns:
                    # Get column type and determine comparison logic
                    dtype = merged.schema[base_col]
                    # Different comparison logic based on data type
                    if str(dtype) in ['Float32', 'Float64']:
                        # For floating point types, treat NULL and NaN as equivalent
                        diff_expr = (
                            # Both are missing (NULL or NaN) - consider equal
                            (pl.col(base_col).is_null() | pl.col(base_col).is_nan())
                            .eq(pl.col(comp_col).is_null() | pl.col(comp_col).is_nan())
                            .not_()
                            # Neither is missing - compare values
                            | ((~pl.col(base_col).is_null() & ~pl.col(base_col).is_nan())
                               & (~pl.col(comp_col).is_null() & ~pl.col(comp_col).is_nan())
                               & (pl.col(base_col) != pl.col(comp_col)))
                        )
                    else:
                        # For all other types (including strings), use null-aware comparison
                        diff_expr = (
                            # Both null - consider equal
                            (pl.col(base_col).is_null() & pl.col(comp_col).is_null()).not_()
                            # One is null, other isn't - consider different
                            & ((pl.col(base_col).is_null() & pl.col(comp_col).is_not_null()) |
                               (pl.col(base_col).is_not_null() & pl.col(comp_col).is_null()) |
                               # Neither is null - compare values
                               (pl.col(base_col).is_not_null() & pl.col(comp_col).is_not_null() &
                                (pl.col(base_col) != pl.col(comp_col))))
                        )

                    diff_rows = merged.filter(diff_expr)

                    if len(diff_rows) > 0:
                        chunk_stats[col] = self._calculate_column_stats(
                            diff_rows, base_col, comp_col
                        )

        return chunk_stats

    def _combine_chunk_results(self, chunk_results: List[Dict]) -> Dict:
        """Combine results from multiple chunks."""
        combined_stats = {}
        for chunk_stat in chunk_results:
            if not chunk_stat:  # Skip empty chunks
                continue

            for col, stats in chunk_stat.items():
                if col not in combined_stats:
                    combined_stats[col] = {
                        "n_differences": 0,
                        "first_n_differences": [],
                        "max_diff": float('-inf') if "max_diff" in stats else None,
                        "mean_diff": 0 if "mean_diff" in stats else None
                    }

                combined_stats[col]["n_differences"] += stats["n_differences"]

                # Keep only first 20 differences across all chunks
                if stats["first_n_differences"]:
                    current_diffs = combined_stats[col]["first_n_differences"]
                    current_diffs.extend(stats["first_n_differences"])
                    combined_stats[col]["first_n_differences"] = current_diffs[:20]

                if "max_diff" in stats and stats["max_diff"] is not None:
                    current_max = combined_stats[col]["max_diff"]
                    if current_max is not None:
                        combined_stats[col]["max_diff"] = max(current_max, stats["max_diff"])
                    else:
                        combined_stats[col]["max_diff"] = stats["max_diff"]

                if "mean_diff" in stats and stats["mean_diff"] is not None:
                    combined_stats[col]["mean_diff"] += (
                        stats["mean_diff"] * stats["n_differences"]
                    )

        # Calculate final means
        for col in combined_stats:
            if combined_stats[col]["mean_diff"] is not None:
                combined_stats[col]["mean_diff"] /= combined_stats[col]["n_differences"]

        return combined_stats

    def _compare_values(self) -> Tuple[Dict, int]:
        """Compare values between DataFrames."""
        # For very large datasets, use streaming and chunks
        if len(self.base_df) > self.chunk_size or len(self.compare_df) > self.chunk_size:
            # Convert to LazyFrames for streaming
            base_lazy = self.base_df.lazy()
            compare_lazy = self.compare_df.lazy()

            # Create chunks using streaming windows
            chunks = []
            max_rows = max(len(self.base_df), len(self.compare_df))

            for i in range(0, max_rows, self.chunk_size):
                end_idx = min(i + self.chunk_size, max_rows)
                base_end = min(end_idx, len(self.base_df))
                compare_end = min(end_idx, len(self.compare_df))

                base_chunk = base_lazy.slice(i, base_end - i) if i < len(self.base_df) else pl.DataFrame().lazy()
                compare_chunk = compare_lazy.slice(i, compare_end - i) if i < len(self.compare_df) else pl.DataFrame().lazy()
                chunks.append((base_chunk, compare_chunk))

            # Process chunks in parallel with dynamic batch size
            n_chunks = len(chunks)
            batch_size = max(1, min(n_chunks // self.n_workers, 4))  # Process 1-4 chunks per worker

            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for i in range(0, n_chunks, batch_size):
                    batch = chunks[i:i + batch_size]
                    future = executor.submit(self._process_chunk_batch, batch)
                    futures.append(future)

                # Collect results as they complete
                chunk_results = []
                for future in futures:
                    chunk_results.extend(future.result())

            # Combine chunk results
            stats = self._combine_chunk_results(chunk_results)
        else:
            # For smaller datasets, process all at once
            stats = self._process_chunk(self.base_df, self.compare_df)

        # Update row statistics
        if self.key_columns:
            base_keys = set(self.base_df.select(self.key_columns).rows())
            compare_keys = set(self.compare_df.select(self.key_columns).rows())
            self.results.structure_results["matched_rows"] = len(base_keys & compare_keys)
            self.results.structure_results["base_only_rows"] = len(base_keys - compare_keys)
            self.results.structure_results["compare_only_rows"] = len(compare_keys - base_keys)
        else:
            min_rows = min(len(self.base_df), len(self.compare_df))
            self.results.structure_results["matched_rows"] = min_rows
            self.results.structure_results["base_only_rows"] = max(0, len(self.base_df) - min_rows)
            self.results.structure_results["compare_only_rows"] = max(0, len(self.compare_df) - min_rows)

        return stats, sum(s["n_differences"] for s in stats.values())

    def compare(self) -> ComparisonResults:
        """Perform the comparison and return results."""
        # Compare structure
        structure_results = self._compare_structure()
        self.results.set_structure_results(structure_results)

        # Compare values
        stats, total_differences = self._compare_values()
        self.results.set_comparison_results(stats, total_differences)

        return self.results
