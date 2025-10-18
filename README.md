# Polars Proc Compare

A high-performance Python implementation of SAS PROC COMPARE functionality using Polars. Optimized for large datasets with efficient memory usage and parallel processing capabilities.

## Features

### Core Functionality
- Structure comparison (columns, types)
- Value-by-value comparison
- Statistical comparison
- HTML report generation
- CSV difference exports
- Jupyter notebook integration

### Performance Optimizations
- Data type normalization for consistent comparisons
- Hash-based row matching for efficiency
- Chunked processing for large datasets
- Parallel processing using multiple CPU cores
- Memory-efficient LazyFrame operations
- Vectorized operations for fast comparisons
- Cached row hashes for repeated operations

## Installation

```bash
poetry install
```

## Performance Configuration

The comparison engine can be tuned for different dataset sizes:

```python
# For very large datasets, adjust chunk size
dc = DataCompare(
    base_df,
    compare_df,
    key_columns=["id"],
    chunk_size=500_000  # Adjust based on available memory
)
```

## Usage

```python
from polars_proc_compare import DataCompare

# Create comparison object
dc = DataCompare(base_df, compare_df)

# Run comparison
results = dc.compare()

# Generate HTML report
results.to_html("comparison_report.html")

# Export differences to CSV
results.to_csv("examples/data/differences.csv")
```

## Project Structure

```
.
├── benchmarks/        # Performance benchmarking scripts and results
├── profiling/        # Profiling scripts and analysis
│   └── results/      # Profiling output files
├── examples/         # Example usage and notebooks
│   └── data/        # Example data files and outputs
└── tests/           # Test suite
    └── utils/       # Test utilities and helpers
```

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .
```
