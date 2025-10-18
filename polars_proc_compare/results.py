from typing import Dict, List
import polars as pl
from pathlib import Path
import jinja2

class ComparisonResults:
    def __init__(self):
        """Initialize the results container."""
        self.structure_results = {}
        self.statistics = {}
        self.total_differences = 0

    def set_structure_results(self, results: Dict):
        """Set the structure comparison results."""
        self.structure_results = results

    def set_comparison_results(self, stats: Dict, total_differences: int):
        """Set the comparison results."""
        self.statistics = stats
        self.total_differences = total_differences

    def to_html(self, output_path: str):
        """Generate HTML report in SAS PROC COMPARE style."""
        template = """
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
Base                   {{ "%6d"|format(structure.base_nrows) }}          {{ "%6d"|format(structure.base_ncols) }}
Compare               {{ "%6d"|format(structure.compare_nrows) }}          {{ "%6d"|format(structure.compare_ncols) }}
                </div>
            </div>
            
            <div class="section">
                <div class="header">Variables Summary</div>
                <div class="content">
Number of Variables in Common: {{ structure.common_cols|length }}
Number of Variables in Base Only: {{ structure.base_only|length }}
Number of Variables in Compare Only: {{ structure.compare_only|length }}

{% if structure.base_only %}Variables in Base Only:
{% for col in structure.base_only %}    {{ col }}
{% endfor %}{% endif %}
{% if structure.compare_only %}Variables in Compare Only:
{% for col in structure.compare_only %}    {{ col }}
{% endfor %}{% endif %}
                </div>
            </div>

            <div class="section">
                <div class="header">Observation Summary</div>
                <div class="content">
Observations in Base:           {{ "%6d"|format(structure.base_nrows) }}
Observations in Compare:       {{ "%6d"|format(structure.compare_nrows) }}
Number of Observations in Common: {{ "%6d"|format(structure.matched_rows) }}
Number of Observations in Base Only: {{ "%6d"|format(structure.base_only_rows) }}
Number of Observations in Compare Only: {{ "%6d"|format(structure.compare_only_rows) }}
                </div>
            </div>

            <div class="section">
                <div class="header">Values Comparison Summary</div>
                <div class="content">
Number of Variables with Differences: {{ statistics|length }}
Total Number of Differences: {{ total_differences }}
                </div>
            </div>

            <div class="section">
                <div class="header">Value Comparison Results</div>
                <div class="content">
{% for col, stats in statistics.items() %}
Variable: {{ col }}    Type: {{ structure.variable_types[col] }}
    Number of Differences: {{ stats.n_differences }}
    {% if stats.max_diff is defined %}Maximum Absolute Difference: {{ "%.6f"|format(stats.max_diff) }}
    {% endif %}{% if stats.mean_diff is defined %}Mean Difference: {{ "%.6f"|format(stats.mean_diff) }}{% endif %}

    First {{ stats.first_n_differences|length }} Difference(s):
       Obs#        Base Value        Compare Value        Difference       % Difference
    {% for diff in stats.first_n_differences %}    {{ "%6d"|format(diff.obs) }}        {{ "%-16s"|format(diff.base) }}    {{ "%-16s"|format(diff.compare) }}    {% if diff.abs_diff is defined %}{{ "%12.6f"|format(diff.abs_diff) }}    {% if diff.pct_diff is defined %}{{ "%12.2f"|format(diff.pct_diff) }}%{% else %}            -{% endif %}{% else %}            -            -{% endif %}
    {% endfor %}
{% endfor %}
                </div>
            </div>
        </body>
        </html>
        """

        env = jinja2.Environment()
        template = env.from_string(template)
        html_content = template.render(
            structure=self.structure_results,
            statistics=self.statistics,
            total_differences=self.total_differences
        )

        Path(output_path).write_text(html_content)

    def to_csv(self, output_path: str):
        """Export differences to CSV."""
        if not self.statistics:
            return

        # Create a list to store all differences
        rows = []
        for col, stats in self.statistics.items():
            for diff in stats["first_n_differences"]:
                row_data = {
                    "Variable": col,
                    "Observation": diff["obs"],
                    "Base_Value": diff["base"],
                    "Compare_Value": diff["compare"],
                    "Difference": diff.get("abs_diff", None),
                    "Pct_Difference": diff.get("pct_diff", None)
                }
                rows.append(row_data)

        # Convert to Polars DataFrame and write to CSV
        if rows:
            df = pl.DataFrame(rows)
            df.write_csv(output_path)