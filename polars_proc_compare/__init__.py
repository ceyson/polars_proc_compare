"""Polars DataFrame comparison utilities."""

from .comparison_engine import DataCompare
from .results import ComparisonResults
from .memory_optimizer import MemoryOptimizedSchema
from .disk_compare import DiskDataCompare

__version__ = "0.1.0"
__all__ = ['DataCompare', 'ComparisonResults', 'MemoryOptimizedSchema', 'DiskDataCompare']
