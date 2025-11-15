"""Polars DataFrame comparison utilities."""

from polars_proc_compare.comparison_engine import DataCompare
from polars_proc_compare.disk_compare import DiskDataCompare
from polars_proc_compare.memory_optimizer import MemoryOptimizedSchema
from polars_proc_compare.results import ComparisonResults
from polars_proc_compare.batch_utils import compare_in_batches, get_memory_usage

__version__ = "0.1.0"
__all__ = [
    "DataCompare",
    "ComparisonResults",
    "MemoryOptimizedSchema",
    "DiskDataCompare",
    "compare_in_batches",
    "get_memory_usage",
]
