"""Tests for type validation functionality."""

import polars as pl
import pytest
from polars_proc_compare.disk_compare import TypeSupport, DiskDataCompare


def test_type_support_validation():
    """Test type validation functionality."""
    # Create a DataFrame with supported types
    supported_df = pl.DataFrame({
        "int64_col": [1, 2, 3],
        "int32_col": pl.Series("int32_col", [1, 2, 3], dtype=pl.Int32),
        "float64_col": [1.0, 2.0, 3.0],
        "float32_col": pl.Series("float32_col", [1.0, 2.0, 3.0], dtype=pl.Float32),
        "string_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
        "datetime_col": pl.date_range(0, 2, interval="1d")
    })

    # Create a DataFrame with unsupported types
    unsupported_df = pl.DataFrame({
        "uint8_col": pl.Series("uint8_col", [1, 2, 3], dtype=pl.UInt8),
        "int8_col": pl.Series("int8_col", [1, 2, 3], dtype=pl.Int8),
        "date_col": pl.Series("date_col", ["2021-01-01", "2021-01-02", "2021-01-03"])
            .cast(pl.Date)
    })

    # Test supported types
    unsupported = TypeSupport.validate_schema(supported_df)
    assert len(unsupported) == 0, "Found unsupported types in supported DataFrame"

    # Test unsupported types
    unsupported = TypeSupport.validate_schema(unsupported_df)
    assert len(unsupported) == 3, "Expected 3 unsupported types"
    assert ("uint8_col", pl.UInt8) in unsupported
    assert ("int8_col", pl.Int8) in unsupported
    assert ("date_col", pl.Date) in unsupported


def test_type_warnings():
    """Test that appropriate warnings are raised."""
    # Create a DataFrame with planned and unplanned types
    df = pl.DataFrame({
        "uint8_col": pl.Series("uint8_col", [1, 2, 3], dtype=pl.UInt8),  # Planned
        "duration_col": pl.Series("duration_col", [1, 2, 3], dtype=pl.Duration),  # Planned
        "list_col": pl.Series("list_col", [[1], [2], [3]])  # Unplanned
    })

    # Test warnings
    TypeSupport.validate_schema(df)

    # Should raise FutureWarning for planned types
    with pytest.warns(FutureWarning) as planned_warnings:
        TypeSupport.warn_unsupported([("uint8_col", pl.UInt8)])
    assert len(planned_warnings) == 1
    assert "planned for version" in str(planned_warnings[0].message)

    # Should raise UserWarning for unplanned types
    with pytest.warns(UserWarning) as user_warnings:
        TypeSupport.warn_unsupported([("list_col", pl.List)])
    assert len(user_warnings) == 1
    assert "may be unreliable" in str(user_warnings[0].message)


def test_ignore_type_warnings():
    """Test that warnings can be suppressed."""
    df = pl.DataFrame({
        "uint8_col": pl.Series("uint8_col", [1, 2, 3], dtype=pl.UInt8)
    })

    # Should not raise warnings when ignore_type_warnings is True
    DiskDataCompare(df, df, ignore_type_warnings=True)

    # Should raise warnings when ignore_type_warnings is False
    with pytest.warns(FutureWarning):
        DiskDataCompare(df, df, ignore_type_warnings=False)
