# Polars Proc Compare

A high-performance Python implementation of SAS PROC COMPARE functionality using Polars. This library provides efficient dataset comparison capabilities with support for large-scale data processing, missing value handling, and detailed difference reporting.

## Features

### Core Functionality
- **Structure Comparison**: Analyzes differences in column names, data types, and dataset dimensions
- **Value Comparison**: Performs element-by-element comparison with support for:
  - Missing values (NULL and NaN)
  - Numeric differences (absolute and percentage)
  - String comparisons
- **Flexible Key Matching**: Compare datasets using key columns or position-based matching
- **Comprehensive Reporting**:
  - HTML reports in SAS PROC COMPARE style (shows first 20 differences per column)
  - CSV exports with complete difference details (all rows)

### Performance Features
- Chunked processing for large datasets
- Parallel execution using ThreadPoolExecutor
- Memory-efficient operations using Polars LazyFrames
- Dynamic chunk sizing based on dataset characteristics
- Configurable memory limits and worker counts
- Column-batch processing for very wide datasets

## Installation

```bash
pip install polars-proc-compare
```

## Usage

### Basic Comparison
```python
from polars_proc_compare import DataCompare
import polars as pl

# Create sample dataframes
base_df = pl.DataFrame(...)
compare_df = pl.DataFrame(...)

# Initialize comparison
dc = DataCompare(base_df, compare_df)

# Run comparison
results = dc.compare()

# Generate reports
results.to_html('comparison_report.html')
results.to_csv('differences.csv')
```

### Advanced Configuration
```python
# Compare with key columns and performance tuning
dc = DataCompare(
    base_df=base_df,
    compare_df=compare_df,
    key_columns=["id", "date"],      # Columns to use for matching rows
    chunk_size=100_000,             # Process data in chunks
    n_workers=8,                    # Number of parallel workers
    memory_limit=1024,             # Memory limit in MB
    use_streaming=True             # Enable streaming for large datasets
)
```

### Batch Processing for Very Wide Datasets
For datasets with many columns (e.g., 100+), you can use batch processing to reduce memory usage:

```python
from polars_proc_compare.batch_utils import compare_in_batches
from pathlib import Path

# Set up temporary directory (optional)
tmp_dir = Path("path/to/your/temp/directory")
tmp_dir.mkdir(parents=True, exist_ok=True)

# Process large datasets in column batches
results = compare_in_batches(
    base_path="path/to/base.parquet",
    compare_path="path/to/compare.parquet",
    key_columns=["id"],
    batch_size=10,           # Process 10 columns at a time
    chunk_size=2_000,        # Process 2k rows at a time
    n_workers=1,             # Single worker to minimize memory
    max_memory_usage=512,    # 512MB memory limit per batch
    max_memory_percent=75.0, # Run GC at 75% memory usage
    monitor_memory=True,     # Enable memory monitoring
    verbose=True,           # Show progress messages
    temp_dir=tmp_dir        # Optional: Use specific temp directory
)

# Generate the same reports as regular comparison
results.to_html("comparison.html")
results.display_html()  # For Jupyter notebooks
```

Batch processing features:
- Process columns in small batches to reduce memory usage
- Monitor and manage memory usage with garbage collection
- Show detailed progress as batches are processed
- Generate the same HTML reports as regular comparison
- Support for both file saving and notebook display
- Optional temporary directory specification for sensitive data

### Example Usage with Large Datasets

```python
from polars_proc_compare import DataCompare
from polars_proc_compare.batch_utils import compare_in_batches
from pathlib import Path
import polars as pl

# Set up directories
output_dir = Path("path/to/output")
temp_dir = Path("path/to/temp")
output_dir.mkdir(parents=True, exist_ok=True)
temp_dir.mkdir(parents=True, exist_ok=True)

# Configure timestamp for output files
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Run comparison with batch processing
results = compare_in_batches(
    base_path="path/to/base.parquet",
    compare_path="path/to/compare.parquet",
    key_columns=["id"],
    batch_size=10,           # Process 10 columns at a time
    chunk_size=2_000,        # Process 2k rows at a time
    n_workers=1,             # Single worker to minimize memory
    max_memory_usage=512,    # 512MB memory limit per batch
    max_memory_percent=75.0, # Run GC at 75% memory usage
    monitor_memory=True,     # Enable memory monitoring
    verbose=True,           # Show progress messages
    temp_dir=temp_dir       # Use specific temp directory
)

# Save reports with timestamp
html_path = output_dir / f"comparison_report_{timestamp}.html"
csv_path = output_dir / f"differences_{timestamp}.csv"

results.to_html(html_path)
results.to_csv(csv_path)

# Display results in notebook (if in notebook environment)
results.display_html()
```

### Performance Tips

1. **Memory Management**:
   - Use batch processing for wide datasets (100+ columns)
   - Enable memory monitoring to track usage
   - Adjust batch and chunk sizes based on available memory
   - Use a dedicated temp directory for better control

2. **File Formats**:
   - Prefer Parquet format for large files
   - Use column-based file formats for efficient column access
   - Consider partitioning very large datasets

3. **Processing Configuration**:
   - Start with single worker (`n_workers=1`) for predictable memory usage
   - Increase workers gradually if more performance is needed
   - Monitor memory usage and adjust `max_memory_usage` accordingly
   - Use smaller batch sizes for very wide datasets

4. **Storage Management**:
   - Use timestamps in filenames for version tracking
   - Clean up temporary files when no longer needed
   - Monitor disk space in temp directory
   - Use appropriate permissions for sensitive data

## Output Format

### CSV Output Columns
- `Variable`: Column name where difference was found
- `Observation`: Row number in base dataset
- `Base_Value`: Value from base dataset
- `Compare_Value`: Value from comparison dataset
- `Difference`: Absolute difference (numeric columns)
- `Pct_Difference`: Percentage difference (numeric columns)

### HTML Report Sections
1. Dataset Summary (row and column counts)
2. Variables Summary (common and unique columns)
3. Observation Summary (matched and unmatched rows)
4. Values Comparison Summary (differences by column)

## Performance Considerations

- **Memory Usage**: The library automatically adjusts chunk size based on available memory
- **Parallelization**: Processes multiple chunks concurrently for better performance
- **Streaming**: Handles large datasets efficiently using Polars' lazy evaluation
- **Type Handling**: Preserves original data types while ensuring consistent comparisons

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Known Limitations

### Type Support
- **Complex Types**: Limited support for List and Struct types
  - Differences are detected but detailed comparisons (e.g., element-by-element for lists) are not yet implemented
  - Warning messages are displayed when comparing these types

### Categorical Data
- **Encoding**: When comparing categorical columns with different encodings, re-encoding is required which may impact performance
- **Memory Usage**: Categorical comparisons might use more memory due to encoding differences

### Performance
- **Memory Optimization**: While the library implements memory-efficient operations, very large categorical columns might require additional memory during comparison
- **Temporary Storage**: Disk-based comparison requires temporary storage space for intermediate files

## Future Enhancements

### Planned Features
1. **Enhanced Type Support**
   - Full support for List type comparisons with element-by-element analysis
   - Struct type comparison with field-level difference reporting
   - Support for more date/time types and formats

2. **Performance Improvements**
   - Optimized categorical comparison without re-encoding
   - Improved memory management for very large datasets
   - GPU acceleration for numeric comparisons

3. **Additional Features**
   - Custom comparison functions for specific column types
   - More flexible tolerance settings for different numeric types
   - Enhanced reporting options with customizable templates
   - Interactive HTML reports with filtering and sorting

4. **Integration Features**
   - Direct database comparison support
   - Cloud storage integration (S3, GCS, etc.)
   - Streaming comparison for real-time data

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .
```
