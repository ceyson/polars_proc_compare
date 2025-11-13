"""Parquet file management for disk-based operations."""

import polars as pl
from pathlib import Path
import tempfile
import shutil
from typing import Optional, Union, Tuple
import uuid

class ParquetManager:
    """Manages parquet files for disk-based operations."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize parquet manager.
        
        Args:
            temp_dir: Optional directory for temporary files. If None, system temp dir is used.
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "polars_compare"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._managed_files = set()
        
    def __del__(self):
        """Cleanup temporary files on deletion."""
        self.cleanup()
        
    def cleanup(self):
        """Remove all managed temporary files."""
        for file in self._managed_files:
            try:
                Path(file).unlink()
            except FileNotFoundError:
                pass
        self._managed_files.clear()
        
        try:
            shutil.rmtree(self.temp_dir)
        except FileNotFoundError:
            pass
            
    def dataframe_to_parquet(self, 
                            df: pl.DataFrame,
                            key_columns: Optional[list] = None) -> str:
        """Convert DataFrame to parquet with optional sorting by key columns.
        
        Args:
            df: DataFrame to convert
            key_columns: Optional columns to sort by before writing
            
        Returns:
            Path to the created parquet file
        """
        file_path = self.temp_dir / f"{uuid.uuid4()}.parquet"
        
        # Sort if key columns provided
        if key_columns:
            df = df.sort(key_columns)
            
        df.write_parquet(file_path)
        self._managed_files.add(str(file_path))
        return str(file_path)
        
    def get_scanner(self, 
                   path: Union[str, Path],
                   columns: Optional[list] = None,
                   predicate: Optional[str] = None) -> pl.LazyFrame:
        """Create a scanner for reading parquet file.
        
        Args:
            path: Path to parquet file
            columns: Optional columns to read
            predicate: Optional predicate pushdown expression
            
        Returns:
            LazyFrame scanner
        """
        return pl.scan_parquet(
            path,
            columns=columns,
            predicate=predicate if predicate else None
        )
        
    def estimate_row_size(self, path: Union[str, Path]) -> float:
        """Estimate average row size in bytes for a parquet file.
        
        Args:
            path: Path to parquet file
            
        Returns:
            Estimated size per row in bytes
        """
        file_size = Path(path).stat().st_size
        metadata = pl.read_parquet_schema(path)
        row_count = pl.scan_parquet(path).select(pl.count()).collect().item()
        return file_size / row_count if row_count > 0 else 0
        
    def get_row_count(self, path: Union[str, Path]) -> int:
        """Get total number of rows in parquet file.
        
        Args:
            path: Path to parquet file
            
        Returns:
            Number of rows
        """
        return pl.scan_parquet(path).select(pl.count()).collect().item()
        
    def create_chunks(self, 
                     path: Union[str, Path],
                     chunk_size: int) -> pl.LazyFrame:
        """Create a chunked scanner for the parquet file.
        
        Args:
            path: Path to parquet file
            chunk_size: Number of rows per chunk
            
        Returns:
            LazyFrame scanner with chunking
        """
        # In Polars 0.19.19, we use scan_parquet with row_count_name
        # This gives us a way to track row positions
        return pl.scan_parquet(
            path,
            row_count_name="__row_id",
            row_count_offset=0
        )
                
    def merge_parquet_files(self,
                           paths: list,
                           output_path: Union[str, Path],
                           key_columns: Optional[list] = None) -> str:
        """Merge multiple parquet files into one.
        
        Args:
            paths: List of parquet file paths
            output_path: Path for merged file
            key_columns: Optional columns to sort by
            
        Returns:
            Path to merged file
        """
        merged = pl.concat([pl.scan_parquet(p) for p in paths]).collect()
        if key_columns:
            merged = merged.sort(key_columns)
        merged.write_parquet(output_path)
        return str(output_path)
