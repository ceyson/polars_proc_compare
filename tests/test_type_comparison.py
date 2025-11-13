"""Tests for type-specific comparison functionality."""

import polars as pl
import pandas as pd
import numpy as np
from decimal import Decimal
import pytest
from polars_proc_compare.disk_compare import TypeComparison


def test_numeric_comparison():
    """Test numeric comparison with various types and edge cases."""
    tc = TypeComparison()
    
    # Basic integer comparison
    assert tc.compare_numeric(1, 1) == (False, None, None)
    assert tc.compare_numeric(1, 2) == (True, 1.0, 100.0)
    
    # Floating point comparison with tolerance
    assert tc.compare_numeric(1.0, 1.0 + 1e-11) == (False, None, None)
    assert tc.compare_numeric(1.0, 1.1) == (True, 0.1, 10.0)
    
    # Decimal comparison
    assert tc.compare_numeric(Decimal('1.0'), Decimal('1.0')) == (False, None, None)
    assert tc.compare_numeric(Decimal('1.0'), Decimal('1.1')) == (True, 0.1, 10.0)
    
    # Edge cases
    assert tc.compare_numeric(0, 1) == (True, 1.0, None)  # Division by zero avoided
    assert tc.compare_numeric(None, 1) == (True, None, None)
    assert tc.compare_numeric(1, None) == (True, None, None)
    assert tc.compare_numeric(float('nan'), float('nan')) == (False, None, None)
    assert tc.compare_numeric(float('inf'), float('inf')) == (False, None, None)
    assert tc.compare_numeric(float('-inf'), float('-inf')) == (False, None, None)
    assert tc.compare_numeric(float('inf'), float('-inf')) == (True, None, None)


def test_datetime_comparison():
    """Test datetime comparison with various formats and timezones."""
    tc = TypeComparison()
    
    # Basic datetime comparison
    dt1 = pd.Timestamp('2023-01-01 12:00:00')
    dt2 = pd.Timestamp('2023-01-01 12:00:00')
    assert tc.compare_datetime(dt1, dt2) == (False, None, None)
    
    # Small difference within tolerance
    dt3 = dt1 + pd.Timedelta(microseconds=0.5)
    assert tc.compare_datetime(dt1, dt3) == (False, None, None)
    
    # Difference outside tolerance
    dt4 = dt1 + pd.Timedelta(seconds=1)
    assert tc.compare_datetime(dt1, dt4) == (True, 1.0, None)
    
    # Different timezones
    dt5 = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
    dt6 = pd.Timestamp('2023-01-01 12:00:00', tz='US/Eastern')
    assert tc.compare_datetime(dt5, dt6)[0] is True  # They're different
    
    # Edge cases
    assert tc.compare_datetime(None, dt1) == (True, None, None)
    assert tc.compare_datetime(dt1, None) == (True, None, None)
    assert tc.compare_datetime('invalid', dt1) == (True, None, None)


def test_string_comparison():
    """Test string comparison with various options."""
    tc = TypeComparison()
    
    # Basic string comparison
    assert tc.compare_string('abc', 'abc') == (False, None, None)
    assert tc.compare_string('abc', 'def') == (True, None, None)
    
    # Case sensitivity
    assert tc.compare_string('ABC', 'abc', case_sensitive=True) == (True, None, None)
    assert tc.compare_string('ABC', 'abc', case_sensitive=False) == (False, None, None)
    
    # Edge cases
    assert tc.compare_string(None, 'abc') == (True, None, None)
    assert tc.compare_string('abc', None) == (True, None, None)
    assert tc.compare_string('', '') == (False, None, None)
    assert tc.compare_string(123, '123') == (False, None, None)  # String coercion


def test_list_comparison():
    """Test list comparison functionality."""
    tc = TypeComparison()
    
    # Basic list comparison
    assert tc.compare_list([1, 2, 3], [1, 2, 3]) == (False, 0, None)
    assert tc.compare_list([1, 2], [1, 2, 3]) == (True, 1, None)
    assert tc.compare_list([1, 2, 3], [1, 2]) == (True, -1, None)
    
    # Different order
    assert tc.compare_list([1, 2, 3], [3, 2, 1]) == (True, 0, None)
    
    # Edge cases
    assert tc.compare_list(None, [1, 2, 3]) == (True, None, None)
    assert tc.compare_list([1, 2, 3], None) == (True, None, None)
    assert tc.compare_list([], []) == (False, 0, None)
    
    # Nested lists
    assert tc.compare_list([[1, 2], [3, 4]], [[1, 2], [3, 4]]) == (False, 0, None)
    assert tc.compare_list([[1, 2], [3, 4]], [[1, 2], [3, 5]]) == (True, 0, None)


def test_struct_comparison():
    """Test struct (dict-like) comparison functionality."""
    tc = TypeComparison()
    
    # Basic struct comparison
    assert tc.compare_struct({'a': 1, 'b': 2}, {'a': 1, 'b': 2}) == (False, None, None)
    assert tc.compare_struct({'a': 1, 'b': 2}, {'a': 1, 'b': 3}) == (True, ['b'], None)
    
    # Different keys
    assert tc.compare_struct({'a': 1}, {'b': 1})[0:2] == (True, ['a', 'b'])
    assert tc.compare_struct({'a': 1, 'b': 2}, {'a': 1})[0:2] == (True, ['b'])
    
    # Edge cases
    assert tc.compare_struct(None, {'a': 1}) == (True, None, None)
    assert tc.compare_struct({'a': 1}, None) == (True, None, None)
    assert tc.compare_struct({}, {}) == (False, None, None)
    
    # Nested structs
    assert tc.compare_struct(
        {'a': {'x': 1}, 'b': {'y': 2}},
        {'a': {'x': 1}, 'b': {'y': 2}}
    ) == (False, None, None)
    assert tc.compare_struct(
        {'a': {'x': 1}, 'b': {'y': 2}},
        {'a': {'x': 1}, 'b': {'y': 3}}
    ) == (True, ['b'], None)
