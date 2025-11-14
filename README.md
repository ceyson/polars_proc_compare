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
    base_df,
    compare_df,
    key_columns=["id", "date"],      # Columns to use for matching rows
    chunk_size=100_000,             # Process data in chunks
    n_workers=8,                    # Number of parallel workers
    memory_limit=1024,             # Memory limit in MB
    use_streaming=True             # Enable streaming for large datasets
)
```

### Using in Databricks

#### Installation and Setup
```python
# Install the package
%pip install polars-proc-compare

# Import libraries
from polars_proc_compare import DataCompare
import polars as pl
```

#### Reading Data
```python
# From Delta tables
base_df = pl.from_pandas(spark.table("base_table").toPandas())
compare_df = pl.from_pandas(spark.table("compare_table").toPandas())

# From Parquet files on volumes
base_df = pl.read_parquet("/dbfs/volumes/my_volume/base.parquet")
compare_df = pl.read_parquet("/dbfs/volumes/my_volume/compare.parquet")

# From Spark DataFrame with schema preservation
from pyspark.sql.types import *

def spark_to_polars(spark_df):
    # Convert complex types and preserve schema
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, (ArrayType, MapType, StructType)):
            spark_df = spark_df.withColumn(field.name, to_json(field.name))
    return pl.from_pandas(spark_df.toPandas())

base_df = spark_to_polars(spark.table("complex_table"))
```

#### Basic Comparison
```python
# Set up paths on mounted volume
volume_path = "/dbfs/volumes/my_volume/comparisons"
html_path = f"{volume_path}/comparison_report.html"
csv_path = f"{volume_path}/differences.csv"

# Run comparison
dc = DataCompare(base_df, compare_df, key_columns=["id"])
results = dc.compare()

# Save reports to volume
results.to_html(html_path)
results.to_csv(csv_path)

# Display results directly in notebook
results.display_html()
```

#### Large Dataset Comparison

##### Basic Disk-Based Mode
```python
# Configure for large datasets
dc = DataCompare(
    base_df=f"/dbfs/volumes/my_volume/base.parquet",  # Path instead of DataFrame
    compare_df=f"/dbfs/volumes/my_volume/compare.parquet",
    key_columns=["id", "date"],
    disk_mode=True,                # Enable disk-based processing
    optimize_dtypes=True,          # Enable memory optimization
    max_memory_usage=4096,         # Set memory limit (MB)
    temp_dir="/dbfs/volumes/my_volume/temp"  # Use volume for temp files
)

# Run comparison with progress tracking
print("Starting comparison...")
results = dc.compare()
print(f"Found {results.total_differences} differences across {len(results.statistics)} columns")

# Display summary in notebook
results.display_html()
```

##### Batch Processing for Very Large Datasets
```python
from polars_proc_compare.batch_utils import compare_in_batches

# For datasets with many columns (100+), use batch processing
results = compare_in_batches(
    base_path="/dbfs/volumes/my_volume/base.parquet",
    compare_path="/dbfs/volumes/my_volume/compare.parquet",
    key_columns=["id"],
    batch_size=50,                # Process 50 columns at a time
    max_memory_percent=80.0,      # Run GC when memory usage exceeds 80%
    chunk_size=10_000,            # Process 10k rows at a time
    n_workers=4,                  # Use 4 parallel workers
    max_memory_usage=2048,        # 2GB memory limit per batch
    monitor_memory=True,          # Enable memory monitoring
    verbose=True                  # Show progress messages
)

# Generate the same formatted reports
results.to_html("/dbfs/volumes/my_volume/report.html")
results.display_html()
```

The batch processing approach is recommended when:
- Dataset has hundreds of columns
- Available memory is limited
- Processing needs to be more granular
- You need better progress monitoring

Batch processing maintains all features:
- Same HTML report format
- Interactive notebook display
- Complete difference tracking
- Memory-efficient operation

#### Working with Volumes and Paths
```python
# Mount point access
volume_path = "/dbfs/volumes/my_volume/comparisons"

# Create directories if needed
import os
os.makedirs(volume_path, exist_ok=True)

# Save outputs with timestamps
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"{volume_path}/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Save reports
results.to_html(f"{output_dir}/report.html")
results.to_csv(f"{output_dir}/differences.csv")
```

#### Important Notes for Databricks Usage

1. **File Access**:
   - Use `/dbfs` prefix for direct file operations
   - Use regular paths (without `/dbfs`) for Spark operations
   - Mount volumes for persistent storage

2. **Memory Management**:
   - Use `disk_mode=True` for large datasets
   - Enable `optimize_dtypes` to reduce memory usage
   - Monitor notebook memory usage in Spark UI

3. **Performance Tips**:
   - Prefer Parquet over CSV for large files
   - Use appropriate cluster configurations
   - Consider partitioning large datasets

4. **Display and Reports**:
   - HTML reports are interactive in notebooks
   - Reports saved to volumes are accessible via file browser
   - Use timestamps in filenames for version tracking

5. **Data Type Handling**:
   - Convert complex Spark types before comparison
   - Handle timezone-aware timestamps appropriately
   - Consider schema differences between Spark and Polars

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
