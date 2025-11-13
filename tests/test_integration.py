"""Integration tests for the complete comparison pipeline."""

import polars as pl
import tempfile
from datetime import datetime
from polars_proc_compare import DiskDataCompare


def test_mixed_type_comparison():
    """Test comparison with various data types."""
    # Create base DataFrame with mixed types
    base_df = pl.DataFrame({
        "id": range(10),
        "int_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "float_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "str_col": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        "cat_col": pl.Series(["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"]).cast(pl.Categorical),
        "bool_col": [True, False, True, False, True, False, True, False, True, False],
        "list_col": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
        "struct_col": [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5},
                    {"a": 6}, {"a": 7}, {"a": 8}, {"a": 9}, {"a": 10}]
    })

    # Create compare DataFrame with specific differences
    compare_df = base_df.clone()

    # Integer difference
    compare_df = compare_df.with_columns(
        pl.when(pl.col("id") == 1)
        .then(pl.lit(100))
        .otherwise(pl.col("int_col"))
        .alias("int_col")
    )

    # Float difference (within and outside tolerance)
    compare_df = compare_df.with_columns(
        pl.when(pl.col("id") == 2)
        .then(pl.col("float_col") + 1e-11)  # Within tolerance
        .when(pl.col("id") == 3)
        .then(pl.col("float_col") + 0.1)    # Outside tolerance
        .otherwise(pl.col("float_col"))
        .alias("float_col")
    )

    # String difference
    compare_df = compare_df.with_columns(
        pl.when(pl.col("id") == 4)
        .then(pl.lit("MODIFIED"))
        .otherwise(pl.col("str_col"))
        .alias("str_col")
    )

    # Categorical difference
    compare_df = compare_df.with_columns(
        pl.when(pl.col("id") == 5)
        .then(pl.lit("NEW_CATEGORY"))
        .otherwise(pl.col("cat_col"))
        .cast(pl.Categorical)
        .alias("cat_col")
    )

    # Boolean difference
    compare_df = compare_df.with_columns(
        pl.when(pl.col("id") == 6)
        .then(~pl.col("bool_col"))
        .otherwise(pl.col("bool_col"))
        .alias("bool_col")
    )


    # List difference
    compare_df = compare_df.with_columns(
        pl.when(pl.col("id") == 9)
        .then(pl.Series("temp", [[100]]))
        .otherwise(pl.col("list_col"))
        .alias("list_col")
    )

    # Struct difference
    compare_df = compare_df.with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.Series("temp", [{"a": 100}]))
        .otherwise(pl.col("struct_col"))
        .alias("struct_col")
    )

    # Run comparison
    with tempfile.TemporaryDirectory() as temp_dir:
        dc = DiskDataCompare(
            base_df=base_df,
            compare_df=compare_df,
            key_columns=["id"],
            temp_dir=temp_dir
        )
        results = dc.compare()

    assert results.total_differences > 0, "No differences found"

    # Check numeric differences
    int_stats = results.statistics.get("int_col", {})
    assert int_stats.get("n_differences") == 1
    assert int_stats.get("max_diff") == 98  # 100 - 2 (at index 1)

    float_stats = results.statistics.get("float_col", {})
    assert float_stats.get("n_differences") == 1  # Only the 0.1 difference
    assert abs(float_stats.get("max_diff") - 0.1) < 1e-6  # Allow for float32 precision

    # Check string differences
    str_stats = results.statistics.get("str_col", {})
    assert str_stats.get("n_differences") == 1
    cat_stats = results.statistics.get("cat_col", {})
    assert cat_stats.get("n_differences") == 1

    # Check boolean differences
    bool_stats = results.statistics.get("bool_col", {})
    assert bool_stats.get("n_differences") == 1


    # Check list differences
    list_stats = results.statistics.get("list_col", {})
    assert list_stats.get("n_differences") == 1

    # Check struct differences
    struct_stats = results.statistics.get("struct_col", {})
    assert struct_stats.get("n_differences") == 1
