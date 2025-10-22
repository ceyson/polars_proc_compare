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
  - HTML reports in SAS PROC COMPARE style
  - CSV exports of differences with row-level details

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
## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .
```
