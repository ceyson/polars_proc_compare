"""Disk-based comparison engine for Polars DataFrames."""

import polars as pl
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Union, Set, Any
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from warnings import warn


class TypeComparison:
    """Handles type-specific comparison logic."""
    
    @staticmethod
    def compare_numeric(base: Any, compare: Any, tolerance: float = 1e-10) -> Tuple[bool, Optional[float], Optional[float]]:
        """Compare numeric values with tolerance.
        
        Args:
            base: Base value
            compare: Comparison value
            tolerance: Tolerance for floating-point comparison
            
        Returns:
            Tuple of (is_different, absolute_diff, percent_diff)
        """
        if base is None or compare is None:
            return True, None, None
            
        try:
            base_val = float(base)
            comp_val = float(compare)
            
            # Handle special values
            if np.isnan(base_val) and np.isnan(comp_val):
                return False, None, None
                
            if np.isinf(base_val) or np.isinf(comp_val):
                if np.isinf(base_val) and np.isinf(comp_val):
                    if np.sign(base_val) == np.sign(comp_val):
                        return False, None, None
                return True, None, None
                
            abs_diff = abs(comp_val - base_val)
            if abs_diff <= tolerance:
                return False, None, None
                
            # Round differences to avoid floating point precision issues
            abs_diff = round(abs_diff, 10)
            pct_diff = round((abs_diff / abs(base_val) * 100), 10) if base_val != 0 else None
            return True, abs_diff, pct_diff
            
        except (ValueError, TypeError):
            return True, None, None
    
    @staticmethod
    def compare_datetime(base: Any, compare: Any, tolerance: pd.Timedelta = pd.Timedelta(microseconds=1)) -> Tuple[bool, Optional[float], None]:
        """Compare datetime values with tolerance.
        
        Args:
            base: Base value
            compare: Comparison value
            tolerance: Tolerance for datetime comparison
            
        Returns:
            Tuple of (is_different, absolute_diff_seconds, None)
        """
        if base is None or compare is None:
            return True, None, None
            
        try:
            base_ts = pd.Timestamp(base)
            comp_ts = pd.Timestamp(compare)
            diff = abs(comp_ts - base_ts)
            
            if diff <= tolerance:
                return False, None, None
                
            return True, diff.total_seconds(), None
            
        except (ValueError, TypeError):
            return True, None, None
    
    @staticmethod
    def compare_string(base: Any, compare: Any, case_sensitive: bool = True) -> Tuple[bool, None, None]:
        """Compare string values.
        
        Args:
            base: Base value
            compare: Comparison value
            case_sensitive: Whether to perform case-sensitive comparison
            
        Returns:
            Tuple of (is_different, None, None)
        """
        if base is None or compare is None:
            return True, None, None
            
        try:
            base_str = str(base)
            comp_str = str(compare)
            
            if not case_sensitive:
                base_str = base_str.lower()
                comp_str = comp_str.lower()
                
            return base_str != comp_str, None, None
            
        except (ValueError, TypeError):
            return True, None, None
    
    @staticmethod
    def compare_list(base: Any, compare: Any) -> Tuple[bool, Optional[int], None]:
        """Compare list values.
        
        Args:
            base: Base value
            compare: Comparison value
            
        Returns:
            Tuple of (is_different, length_diff, None)
        """
        if base is None or compare is None:
            return True, None, None
            
        try:
            base_list = list(base)
            comp_list = list(compare)
            
            if len(base_list) != len(comp_list):
                return True, len(comp_list) - len(base_list), None
                
            return base_list != comp_list, 0, None
            
        except (ValueError, TypeError):
            return True, None, None
    
    @staticmethod
    def compare_struct(base: Any, compare: Any) -> Tuple[bool, Optional[List[str]], None]:
        """Compare struct (dict-like) values.
        
        Args:
            base: Base value
            compare: Comparison value
            
        Returns:
            Tuple of (is_different, differing_keys, None)
        """
        if base is None or compare is None:
            return True, None, None
            
        try:
            base_dict = dict(base)
            comp_dict = dict(compare)
            
            if base_dict.keys() != comp_dict.keys():
                # Sort keys for consistent order
                diff_keys = sorted(set(comp_dict.keys()) ^ set(base_dict.keys()))
                return True, diff_keys, None
                
            differing_keys = sorted(k for k in base_dict if base_dict[k] != comp_dict[k])
            return bool(differing_keys), differing_keys or None, None
            
        except (ValueError, TypeError):
            return True, None, None

class TypeSupport:
    """Manages supported data types and validation."""
    
    SUPPORTED_TYPES: Set[pl.DataType] = {
        # Numeric types
        pl.Int64, pl.Int32, pl.Float64, pl.Float32,
        pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8,
        pl.Int16, pl.Int8,
        pl.Decimal,
        
        # String types
        pl.Utf8, pl.Categorical,
        
        # Boolean type
        pl.Boolean,
        
        # Date and time types
        pl.Datetime, pl.Date, pl.Time, pl.Duration,
        
        # Special types
        pl.Object,  # Required for some datetime operations
        pl.List,    # For array-like columns
        pl.Struct   # For nested data structures
    }
    
    # Configuration for type-specific comparisons
    NUMERIC_TOLERANCE = 1e-10  # Default tolerance for floating-point comparisons
    DATETIME_TOLERANCE = pd.Timedelta(microseconds=1)  # Default tolerance for datetime comparisons
    
    # Type categories for specialized comparison logic
    NUMERIC_TYPES = {
        pl.Int64, pl.Int32, pl.Int16, pl.Int8,
        pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8,
        pl.Float64, pl.Float32, pl.Decimal
    }
    
    DATETIME_TYPES = {
        pl.Datetime, pl.Date, pl.Time, pl.Duration
    }
    
    STRING_TYPES = {
        pl.Utf8, pl.Categorical
    }
    
    COMPLEX_TYPES = {
        pl.List, pl.Struct
    }
    
    @classmethod
    def validate_schema(cls, df: pl.DataFrame) -> List[Tuple[str, pl.DataType]]:
        """Validate DataFrame schema against supported types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of (column_name, dtype) tuples for unsupported columns
        """
        unsupported = []
        for col, dtype in df.schema.items():
            if dtype not in cls.SUPPORTED_TYPES:
                unsupported.append((col, dtype))
        return unsupported
    
    @classmethod
    def warn_unsupported(cls, unsupported: List[Tuple[str, pl.DataType]]):
        """Generate appropriate warnings for unsupported types."""
        for col, dtype in unsupported:
            if dtype in cls.PLANNED_TYPES:
                warn(
                    f"Column '{col}' has type {dtype} which is not yet supported. "
                    f"Support planned for version {cls.PLANNED_TYPES[dtype]}.",
                    FutureWarning
                )
            else:
                warn(
                    f"Column '{col}' has unsupported type {dtype}. "
                    "Comparison results may be unreliable.",
                    UserWarning
                )

from .memory_optimizer import MemoryOptimizedSchema, MemoryMonitor
from .parquet_manager import ParquetManager
from .results import ComparisonResults

class DiskDataCompare:
    """Disk-based comparison engine for large datasets."""
    
    def __init__(
        self,
        base_df: Union[pl.DataFrame, str, Path],
        compare_df: Union[pl.DataFrame, str, Path],
        key_columns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        n_workers: Optional[int] = None,
        max_memory_usage: Optional[int] = None,
        temp_dir: Optional[str] = None,
        optimize_dtypes: bool = True,
        schema_optimizer: Optional[MemoryOptimizedSchema] = None,
        ignore_type_warnings: bool = False
    ):
        """Initialize disk-based comparison engine.
        
        Args:
            base_df: Base DataFrame or path to parquet file
            compare_df: Comparison DataFrame or path to parquet file
            key_columns: Columns to use as keys for matching rows
            chunk_size: Number of rows per chunk
            n_workers: Number of worker threads
            max_memory_usage: Maximum memory usage in MB
            temp_dir: Directory for temporary files
            optimize_dtypes: Whether to optimize data types
            schema_optimizer: Custom schema optimizer
            ignore_type_warnings: Whether to suppress warnings about unsupported types
        """
        self.key_columns = key_columns
        self.n_workers = n_workers or min(32, mp.cpu_count() * 2)
        self.optimize_dtypes = optimize_dtypes
        self.schema_optimizer = schema_optimizer or MemoryOptimizedSchema()
        self.memory_monitor = MemoryMonitor(max_memory_usage)
        self.parquet_manager = ParquetManager(temp_dir)
        self.results = ComparisonResults()
        
        # Validate types if input is DataFrame
        if not ignore_type_warnings:
            if isinstance(base_df, pl.DataFrame):
                unsupported = TypeSupport.validate_schema(base_df)
                if unsupported:
                    TypeSupport.warn_unsupported(unsupported)
            
            if isinstance(compare_df, pl.DataFrame):
                unsupported = TypeSupport.validate_schema(compare_df)
                if unsupported:
                    TypeSupport.warn_unsupported(unsupported)
        
        # Convert inputs to parquet if needed
        self.base_path = (base_df if isinstance(base_df, (str, Path))
                         else self.parquet_manager.dataframe_to_parquet(base_df, key_columns))
        self.compare_path = (compare_df if isinstance(compare_df, (str, Path))
                           else self.parquet_manager.dataframe_to_parquet(compare_df, key_columns))
        
        # Calculate optimal chunk size
        if chunk_size is None:
            avg_row_size = max(
                self.parquet_manager.estimate_row_size(self.base_path),
                self.parquet_manager.estimate_row_size(self.compare_path)
            )
            total_rows = max(
                self.parquet_manager.get_row_count(self.base_path),
                self.parquet_manager.get_row_count(self.compare_path)
            )
            self.chunk_size = self.memory_monitor.calculate_safe_chunk_size(
                total_rows, avg_row_size
            )
        else:
            self.chunk_size = chunk_size

    def _optimize_chunk(self, chunk: pl.DataFrame) -> pl.DataFrame:
        """Optimize chunk dtypes if enabled."""
        if self.optimize_dtypes:
            return self.schema_optimizer.optimize_chunk(chunk)
        return chunk

    def _compare_structure(self) -> Dict:
        """Compare the structure of both datasets."""
        base_schema = pl.read_parquet_schema(self.base_path)
        compare_schema = pl.read_parquet_schema(self.compare_path)
        
        base_cols = set(base_schema.keys())
        comp_cols = set(compare_schema.keys())
        
        return {
            "common_cols": list(base_cols & comp_cols),
            "base_only": list(base_cols - comp_cols),
            "compare_only": list(comp_cols - base_cols),
            "base_schema": base_schema,
            "compare_schema": compare_schema,
            "base_nrows": self.parquet_manager.get_row_count(self.base_path),
            "compare_nrows": self.parquet_manager.get_row_count(self.compare_path),
            "base_ncols": len(base_cols),
            "compare_ncols": len(comp_cols),
            "variable_types": {col: str(base_schema[col]) for col in base_cols & comp_cols},
            "matched_rows": 0,
            "base_only_rows": 0,
            "compare_only_rows": 0
        }

    def _process_keyed_chunks(self, base_chunk: pl.DataFrame, compare_chunk: pl.DataFrame) -> Dict:
        """Process chunks for key-based comparison."""
        base_data = self._optimize_chunk(base_chunk)
        compare_data = self._optimize_chunk(compare_chunk)
        
        merged = base_data.join(
            compare_data,
            on=self.key_columns,
            how="outer",
            suffix="_compare"
        )
        
        return self._calculate_differences(merged)

    def _process_positional_chunks(self, base_chunk: pl.DataFrame, compare_chunk: pl.DataFrame) -> Dict:
        """Process chunks for position-based comparison."""
        base_data = self._optimize_chunk(base_chunk)
        compare_data = self._optimize_chunk(compare_chunk)
        
        # Add row numbers within chunk
        base_data = base_data.with_row_count("chunk_id")
        compare_data = compare_data.with_row_count("chunk_id")
        
        merged = base_data.join(
            compare_data,
            on="chunk_id",
            how="outer",
            suffix="_compare"
        ).with_row_count("__row_id")
        
        return self._calculate_differences(merged)

    def _calculate_differences(self, merged: pl.DataFrame) -> Dict:
        """Calculate differences between merged chunks."""
        chunk_stats = {}
        for col in self.results.structure_results["common_cols"]:
            if col not in (self.key_columns or []):
                base_col = col
                comp_col = f"{col}_compare"
                
                if comp_col in merged.columns:
                    # Handle different data types appropriately
                    dtype = merged.schema[base_col]
                    if dtype == pl.Categorical:
                        # Convert categoricals to strings for comparison
                        diff_expr = (
                            (pl.col(base_col).cast(pl.Utf8) != pl.col(comp_col).cast(pl.Utf8))
                            | (pl.col(base_col).is_null() & pl.col(comp_col).is_not_null())
                            | (pl.col(base_col).is_not_null() & pl.col(comp_col).is_null())
                        )
                    elif str(dtype) in ['Float32', 'Float64']:
                        diff_expr = (
                            (pl.col(base_col).is_null() | pl.col(base_col).is_nan())
                            .eq(pl.col(comp_col).is_null() | pl.col(comp_col).is_nan())
                            .not_()
                            | ((~pl.col(base_col).is_null() & ~pl.col(base_col).is_nan())
                               & (~pl.col(comp_col).is_null() & ~pl.col(comp_col).is_nan())
                               & (pl.col(base_col) != pl.col(comp_col)))
                        )
                    else:
                        diff_expr = (
                            (pl.col(base_col).is_null() & pl.col(comp_col).is_null()).not_()
                            & ((pl.col(base_col).is_null() & pl.col(comp_col).is_not_null())
                               | (pl.col(base_col).is_not_null() & pl.col(comp_col).is_null())
                               | (pl.col(base_col).is_not_null() & pl.col(comp_col).is_not_null()
                                  & (pl.col(base_col) != pl.col(comp_col))))
                        )
                    
                    diff_rows = merged.filter(diff_expr)
                    if len(diff_rows) > 0:
                        chunk_stats[col] = self._calculate_column_stats(
                            diff_rows, base_col, comp_col
                        )
        
        return chunk_stats

    def _calculate_column_stats(self, diff_rows: pl.DataFrame, base_col: str, comp_col: str) -> Dict:
        """Calculate statistics for a column with differences."""
        col_stats = {
            "n_differences": len(diff_rows),
            "first_n_differences": []
        }
        
        if diff_rows[base_col].dtype.is_numeric():
            # Handle numeric comparisons based on type
            if str(diff_rows[base_col].dtype).startswith('Int'):
                diffs = diff_rows.with_columns([
                    (pl.col(comp_col) - pl.col(base_col)).alias("abs_diff"),
                    (pl.when(pl.col(base_col) != 0)
                     .then(((pl.col(comp_col) - pl.col(base_col)) / pl.col(base_col) * 100).cast(pl.Float64))
                     .otherwise(None)
                     .alias("pct_diff"))
                ])
            else:
                diffs = diff_rows.with_columns([
                    ((pl.col(comp_col) - pl.col(base_col))
                     .round(4)
                     .alias("abs_diff")),
                    (pl.when(pl.col(base_col) != 0)
                     .then(((pl.col(comp_col) - pl.col(base_col)) / pl.col(base_col) * 100).round(2))
                     .otherwise(None)
                     .alias("pct_diff"))
                ])

            # Include row numbers and differences
            select_cols = [
                pl.col("__row_id").alias("obs"),
                pl.col(base_col).alias("base"),
                pl.col(comp_col).alias("compare"),
                pl.col("abs_diff"),
                pl.col("pct_diff")
            ]
            col_stats["first_n_differences"] = diffs.select(select_cols).rows(named=True)
            
            # Calculate overall statistics
            valid_diffs = (
                diff_rows
                .filter(pl.col(base_col).is_not_null() & pl.col(comp_col).is_not_null())
                .select([(pl.col(comp_col) - pl.col(base_col)).abs().alias("diff")])
            )
            
            if len(valid_diffs) > 0:
                col_stats.update({
                    "max_diff": float(valid_diffs["diff"].max()),
                    "mean_diff": float(valid_diffs["diff"].mean())
                })
            else:
                col_stats.update({
                    "max_diff": None,
                    "mean_diff": None
                })
        else:
            # Non-numeric comparisons
            select_cols = [
                pl.col("__row_id").alias("obs"),
                pl.col(base_col).alias("base"),
                pl.col(comp_col).alias("compare")
            ]
            col_stats["first_n_differences"] = diff_rows.select(select_cols).rows(named=True)
        
        return col_stats

    def compare(self) -> ComparisonResults:
        """Perform the comparison and return results."""
        try:
            # Compare structure
            structure_results = self._compare_structure()
            self.results.set_structure_results(structure_results)
            
            # Create scanners with row IDs
            base_scanner = self.parquet_manager.create_chunks(self.base_path, self.chunk_size)
            compare_scanner = self.parquet_manager.create_chunks(self.compare_path, self.chunk_size)
            
            # Process data in chunks
            all_stats = {}
            total_differences = 0
            
            # Collect both datasets with row IDs
            base_data = base_scanner.collect()
            compare_data = compare_scanner.collect()
            
            # Process in chunks
            total_rows = len(base_data)
            chunks = []
            
            for start_idx in range(0, total_rows, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_rows)
                base_chunk = base_data[start_idx:end_idx]
                compare_chunk = compare_data[start_idx:end_idx]
                chunks.append((base_chunk, compare_chunk))
            
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for base_chunk, compare_chunk in chunks:
                    if self.key_columns:
                        future = executor.submit(self._process_keyed_chunks, base_chunk, compare_chunk)
                    else:
                        future = executor.submit(self._process_positional_chunks, base_chunk, compare_chunk)
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    chunk_stats = future.result()
                    for col, stats in chunk_stats.items():
                        if col not in all_stats:
                            all_stats[col] = {
                                "n_differences": 0,
                                "first_n_differences": [],
                                "max_diff": float('-inf'),
                                "mean_diff": 0
                            }
                        
                        all_stats[col]["n_differences"] += stats["n_differences"]
                        all_stats[col]["first_n_differences"].extend(
                            stats["first_n_differences"][:20 - len(all_stats[col]["first_n_differences"])]
                        )
                        
                        if "max_diff" in stats:
                            all_stats[col]["max_diff"] = max(
                                all_stats[col]["max_diff"],
                                stats["max_diff"]
                            )
                        
                        if "mean_diff" in stats:
                            all_stats[col]["mean_diff"] = (
                                (all_stats[col]["mean_diff"] * total_differences +
                                 stats["mean_diff"] * stats["n_differences"]) /
                                (total_differences + stats["n_differences"])
                            )
                        
                        total_differences += stats["n_differences"]
            
            # Update row statistics
            if self.key_columns:
                base_keys = set(base_data.select(self.key_columns).rows())
                compare_keys = set(compare_data.select(self.key_columns).rows())
                self.results.structure_results["matched_rows"] = len(base_keys & compare_keys)
                self.results.structure_results["base_only_rows"] = len(base_keys - compare_keys)
                self.results.structure_results["compare_only_rows"] = len(compare_keys - base_keys)
            else:
                min_rows = min(len(base_data), len(compare_data))
                self.results.structure_results["matched_rows"] = min_rows
                self.results.structure_results["base_only_rows"] = max(0, len(base_data) - min_rows)
                self.results.structure_results["compare_only_rows"] = max(0, len(compare_data) - min_rows)
            
            # Update comparison results
            self.results.set_comparison_results(all_stats, total_differences)
            
            # Add memory optimization stats if enabled
            if self.optimize_dtypes:
                self.results.optimization_stats = self.schema_optimizer.get_optimization_stats()
            
            return self.results
            
        finally:
            # Cleanup temporary files
            self.parquet_manager.cleanup()
