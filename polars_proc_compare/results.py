"""Results handling for DataFrame comparisons."""

import polars as pl
from pathlib import Path
from typing import Dict, Optional
from jinja2 import Template

class ComparisonResults:
    """Handles comparison results and report generation."""

    def __init__(self):
        """Initialize comparison results."""
        self.structure_results = {}
        self.comparison_results = {}
        self.total_differences = 0
        self.optimization_stats = {}

    def set_structure_results(self, results: Dict):
        """Set structure comparison results."""
        self.structure_results = results

    def set_comparison_results(self, stats: Dict, total_differences: int):
        """Set value comparison results."""
        self.comparison_results = stats
        self.total_differences = total_differences

    def to_html(self, output_path: str):
        """Generate HTML report."""
        if not self.structure_results:
            return

        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Comparison Report</title>
            <style>
                body { font-family: monospace; margin: 20px; }
                pre { margin: 0; }
                .section { margin-bottom: 30px; white-space: pre; }
                .header { 
                    font-weight: bold; 
                    margin-top: 20px;
                    border-bottom: 1px solid #000;
                }
                .content {
                    padding-left: 20px;
                    white-space: pre;
                }
            </style>
        </head>
        <body>
            <div class="section">
                <div class="header">Data Set Summary</div>
                <div class="content">
Dataset             Observations    Variables
Base                     {{base_nrows}}               {{base_ncols}}
Compare                 {{compare_nrows}}               {{compare_ncols}}
                </div>
            </div>
            
            <div class="section">
                <div class="header">Variables Summary</div>
                <div class="content">
Number of Variables in Common: {{n_common}}
Number of Variables in Base Only: {{n_base_only}}
Number of Variables in Compare Only: {{n_compare_only}}

{% if base_only %}Base Only Variables:
{% for var in base_only %}    {{var}}
{% endfor %}{% endif %}

{% if compare_only %}Compare Only Variables:
{% for var in compare_only %}    {{var}}
{% endfor %}{% endif %}
                </div>
            </div>

            <div class="section">
                <div class="header">Observation Summary</div>
                <div class="content">
Observations in Base:             {{base_nrows}}
Observations in Compare:         {{compare_nrows}}
Number of Observations in Common:   {{matched_rows}}
Number of Observations in Compare Only:      {{compare_only_rows}}
                </div>
            </div>

            <div class="section">
                <div class="header">Values Comparison Summary:</div>
                <div class="content">
Number of differences found: {{total_differences}}
Number of columns with differences: {{n_vars_with_diffs}}
                </div>
            </div>

            <div class="section">
                <div class="header">Memory Optimization:</div>
                <div class="content">
{% if optimization_stats %}
Original size: {{ (optimization_stats.original_size / (1024 * 1024))|round(2) }} MB
Optimized size: {{ (optimization_stats.optimized_size / (1024 * 1024))|round(2) }} MB
Memory reduction: {{ optimization_stats.reduction_percent|round(1) }}%
{% endif %}
{% for var, stats in statistics.items() %}
Variable: {{var}}    Type: {{variable_types[var]}}
    Number of Differences: {{stats.n_differences}}
    {% if stats.max_diff is not none %}Maximum Absolute Difference: {{stats.max_diff}}
    Mean Difference: {{stats.mean_diff}}{% endif %}

    First {{stats.first_n_differences|length}} Difference(s):
       Obs#        Base Value        Compare Value        Difference       % Difference
{% for diff in stats.first_n_differences %}             {{diff.obs}}        {{diff.base}}                   {{diff.compare}}                      {{diff.abs_diff}}            {{diff.pct_diff}}
{% endfor %}    

{% endfor %}
                </div>
            </div>
        </body>
        </html>
        """)

        html_content = template.render(
            base_nrows=self.structure_results["base_nrows"],
            compare_nrows=self.structure_results["compare_nrows"],
            base_ncols=self.structure_results["base_ncols"],
            compare_ncols=self.structure_results["compare_ncols"],
            n_common=len(self.structure_results["common_cols"]),
            n_base_only=len(self.structure_results["base_only"]),
            n_compare_only=len(self.structure_results["compare_only"]),
            base_only=self.structure_results["base_only"],
            compare_only=self.structure_results["compare_only"],
            matched_rows=self.structure_results["matched_rows"],
            base_only_rows=self.structure_results["base_only_rows"],
            compare_only_rows=self.structure_results["compare_only_rows"],
            n_vars_with_diffs=len(self.comparison_results),
            total_differences=self.total_differences,
            variable_types=self.structure_results["variable_types"],
            statistics=self.statistics
        )

        Path(output_path).write_text(html_content)

    def to_csv(self, output_path: str):
        """Export differences to CSV."""
        if not self.statistics:
            return
            
        # Create a list to store all differences
        rows = []
        for col, stats in self.statistics.items():
            for diff in stats.get("first_n_differences", []):
                # Handle both numeric and non-numeric values
                base_val = diff.get('base')
                comp_val = diff.get('compare')
                abs_diff = diff.get('abs_diff')
                pct_diff = diff.get('pct_diff')
                
                # Format values based on type
                try:
                    base_str = f"{float(base_val):.4f}" if base_val is not None else "NULL"
                except (ValueError, TypeError):
                    base_str = str(base_val) if base_val is not None else "NULL"
                    
                try:
                    comp_str = f"{float(comp_val):.4f}" if comp_val is not None else "NULL"
                except (ValueError, TypeError):
                    comp_str = str(comp_val) if comp_val is not None else "NULL"
                    
                try:
                    diff_str = f"{float(abs_diff):.4f}" if abs_diff is not None else ""
                except (ValueError, TypeError):
                    diff_str = str(abs_diff) if abs_diff is not None else ""
                    
                try:
                    pct_str = f"{float(pct_diff):.2f}" if pct_diff is not None else ""
                except (ValueError, TypeError):
                    pct_str = str(pct_diff) if pct_diff is not None else ""
                
                row_data = {
                    "Variable": str(col),
                    "Observation": str(diff.get("obs", "")),
                    "Base_Value": base_str,
                    "Compare_Value": comp_str,
                    "Difference": diff_str,
                    "Pct_Difference": pct_str
                }
                rows.append(row_data)

        # Convert to Polars DataFrame with explicit schema
        if rows:
            schema = [
                ("Variable", pl.Utf8),
                ("Observation", pl.Utf8),
                ("Base_Value", pl.Utf8),
                ("Compare_Value", pl.Utf8),
                ("Difference", pl.Utf8),
                ("Pct_Difference", pl.Utf8)
            ]
            
            df = pl.DataFrame(rows, schema=schema)
            df.write_csv(output_path)

    @property
    def statistics(self) -> Dict:
        """Get comparison statistics."""
        return self.comparison_results
