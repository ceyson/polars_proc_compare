"""Memory optimization utilities for Polars DataFrames."""

import polars as pl
from typing import Dict, Optional


class MemoryOptimizedSchema:
    """Optimizes DataFrame memory usage by adjusting dtypes."""
    
    def __init__(self):
        """Initialize memory optimizer."""
        self.int_max_values = {}
        self.int_min_values = {}
        self.float_max_abs = {}
        self.string_unique_counts = {}
        self.total_original_size = 0
        self.total_optimized_size = 0
        self.categorical_threshold = 0.5  # Convert to categorical if unique ratio below this
    
    def optimize_integer_dtype(self, series: pl.Series) -> pl.Series:
        """Convert integer series to smallest possible dtype."""
        if series.dtype not in [pl.Int64, pl.UInt64]:
            return series
            
        # Get min/max values
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return series
            
        min_val = non_null.min()
        max_val = non_null.max()
        
        # Determine smallest possible dtype
        if min_val >= 0:
            if max_val <= 255:
                return series.cast(pl.UInt8)
            elif max_val <= 65535:
                return series.cast(pl.UInt16)
            elif max_val <= 4294967295:
                return series.cast(pl.UInt32)
        else:
            if min_val >= -128 and max_val <= 127:
                return series.cast(pl.Int8)
            elif min_val >= -32768 and max_val <= 32767:
                return series.cast(pl.Int16)
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return series.cast(pl.Int32)
        
        return series
    
    def optimize_float_dtype(self, series: pl.Series) -> pl.Series:
        """Convert float series to Float32 if possible."""
        if series.dtype != pl.Float64:
            return series
            
        # Check if we can safely downcast to Float32
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return series
            
        max_abs = non_null.abs().max()
        if max_abs <= 3.4e38:  # Max value for Float32
            return series.cast(pl.Float32)
        
        return series
    
    def optimize_string_dtype(self, series: pl.Series) -> pl.Series:
        """Convert string series to categorical if cardinality is low enough."""
        if not str(series.dtype).startswith('Utf8'):
            return series
            
        # Get unique ratio
        n_unique = series.n_unique()
        n_total = len(series)
        unique_ratio = n_unique / n_total if n_total > 0 else 1.0
        print(f"String column stats: {n_unique} unique values out of {n_total} ({unique_ratio:.2%})")
            
        if unique_ratio <= self.categorical_threshold:
            print(f"Converting to categorical (ratio {unique_ratio:.2%} <= threshold {self.categorical_threshold:.2%})")
            return series.cast(pl.Categorical)
        return series
    
    def optimize_chunk(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimize all columns in a DataFrame chunk."""
        # Track original size
        original_size = df.estimated_size()
        self.total_original_size += original_size
        
        print("\nOriginal schema:")
        for col, dtype in df.schema.items():
            print(f"{col}: {dtype}")
        
        # Optimize columns
        optimized_cols = {}
        for col in df.columns:
            series = df[col]
            orig_dtype = series.dtype
            if series.dtype in [pl.Int64, pl.UInt64]:
                series = self.optimize_integer_dtype(series)
            elif series.dtype == pl.Float64:
                series = self.optimize_float_dtype(series)
            elif series.dtype == pl.Utf8:
                series = self.optimize_string_dtype(series)
            optimized_cols[col] = series
            if series.dtype != orig_dtype:
                print(f"Optimized {col}: {orig_dtype} -> {series.dtype}")
        
        # Create optimized DataFrame and track size
        optimized_df = pl.DataFrame(optimized_cols)
        self.total_optimized_size += optimized_df.estimated_size()
        
        return optimized_df
    
    def get_optimization_stats(self) -> Dict:
        """Get overall optimization statistics."""
        return {
            "original_size": self.total_original_size,
            "optimized_size": self.total_optimized_size,
            "reduction_percent": ((self.total_original_size - self.total_optimized_size) / self.total_original_size * 100) 
                                if self.total_original_size > 0 else 0
        }


class MemoryMonitor:
    """Monitors and manages memory usage."""
    
    def __init__(self, max_memory_mb: Optional[int] = None):
        """Initialize memory monitor."""
        self.max_memory_mb = max_memory_mb
    
    def calculate_safe_chunk_size(self, total_rows: int, avg_row_size: float) -> int:
        """Calculate safe chunk size based on memory constraints."""
        if not self.max_memory_mb:
            return min(100_000, total_rows)  # Default chunk size
            
        # Aim to use at most 1/3 of max memory for each chunk
        safe_memory = (self.max_memory_mb * 1024 * 1024) / 3
        safe_rows = int(safe_memory / avg_row_size)
        
        return min(safe_rows, total_rows, 100_000)  # Cap at 100K rows
