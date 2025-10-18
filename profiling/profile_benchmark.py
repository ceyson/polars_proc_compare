"""Profile the benchmark code to identify bottlenecks."""

import cProfile
import pstats
from line_profiler import LineProfiler
import polars as pl
from polars_proc_compare import DataCompare
from polars_proc_compare.data_generator import create_delta_dataset


def profile_comparison(size: int = 500_000, delta_pct: float = 1.0, chunk_size: int = 50_000):
    """Profile a single comparison run."""
    # Generate test data
    base_df = pl.DataFrame({
        "id": range(1, size + 1),
        **{
            f"num_{i}": pl.Series(f"num_{i}", range(size))
            for i in range(5)
        },
        **{
            f"str_{i}": pl.Series(f"str_{i}", [f"val_{j}" for j in range(size)])
            for i in range(3)
        }
    })

    compare_df, _ = create_delta_dataset(
        base_df,
        delta_percentage=delta_pct,
        seed=42,
        exclude_columns=["id"]
    )

    # Create DataCompare instance
    dc = DataCompare(
        base_df,
        compare_df,
        key_columns=["id"],
        chunk_size=chunk_size
    )

    # Run comparison
    results = dc.compare()
    return results

def main():
    # Profile with cProfile for overall timing
    profiler = cProfile.Profile()
    profiler.enable()

    _ = profile_comparison()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')

    # Save profiling results
    with open('results/profile_results.txt', 'w') as f:
        stats.stream = f
        stats.print_stats()

    # Profile specific functions with line_profiler
    lp = LineProfiler()
    lp.add_function(DataCompare._process_chunk)
    lp.add_function(DataCompare._normalize_column)
    lp.add_function(DataCompare._compare_values)
    lp.add_function(create_delta_dataset)

    lp_wrapper = lp(profile_comparison)
    lp_wrapper()

    # Save line profiling results
    with open('results/line_profile_results.txt', 'w') as f:
        lp.print_stats(stream=f)


if __name__ == "__main__":
    main()
