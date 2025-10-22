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
        self.key_columns = []  # Store key columns

    def set_structure_results(self, results: Dict):
        """Set the structure comparison results."""
        self.structure_results = results
        # Store key columns if present
        self.key_columns = results.get('key_columns', [])

    def set_comparison_results(self, stats: Dict, total_differences: int):
        """Set the comparison results."""
        self.statistics = stats
        self.total_differences = total_differences

    def to_html(self, output_path: str):
        """Generate HTML report in SAS PROC COMPARE style."""
        env = jinja2.Environment()
        env.globals.update({
            'none': None  # Make None available in template
        })
        template = env.from_string("""

        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Comparison Report</title>
            <style>
                body { 
                    font-family: monospace; 
                    margin: 20px; 
                    line-height: 1.5;
                }
                pre { 
                    margin: 0; 
                }
                .section { 
                    margin-bottom: 30px; 
                    white-space: pre; 
                }
                .header { 
                    font-weight: bold; 
                    margin-top: 20px;
                    border-bottom: 1px solid #000;
                    padding-bottom: 5px;
                }
                .content {
                    padding-left: 20px;
                    white-space: pre;
                    font-family: 'Courier New', Courier, monospace;
                }
                .diff-container {
                    margin-left: 20px;
                    font-family: 'Courier New', Courier, monospace;
                }
                .diff-table {
                    border-spacing: 0;
                    border-collapse: collapse;
                    width: 100%;
                }
                .diff-table td, .diff-table th {
                    padding: 0;
                    white-space: pre;
                    text-align: left;
                }
                .diff-table td.number, .diff-table th.number {
                    text-align: right;
                }
                .diff-header {
                    margin-bottom: 5px;
                    border-spacing: 0;
                    width: 100%;
                }
                .diff-header th {
                    font-weight: bold;
                    text-align: left;
                    white-space: pre;
                    padding: 0;
                }
            </style>
        </head>
        <body>
            <div class="section">
                <div class="header">Data Set Summary</div>
                <div class="content">
Dataset             Observations    Variables
Base                   {{ "%6d"|format(structure.base_nrows if structure.base_nrows is defined and structure.base_nrows != none else 0) }}          {{ "%6d"|format(structure.base_ncols if structure.base_ncols is defined and structure.base_ncols != none else 0) }}
Compare               {{ "%6d"|format(structure.compare_nrows if structure.compare_nrows is defined and structure.compare_nrows != none else 0) }}          {{ "%6d"|format(structure.compare_ncols if structure.compare_ncols is defined and structure.compare_ncols != none else 0) }}
                </div>
            </div>
            
            <div class="section">
                <div class="header">Variables Summary</div>
                <div class="content">
Number of Variables in Common: {{ structure.common_cols|length if structure.common_cols is defined and structure.common_cols != none else 0 }}
Number of Variables in Base Only: {{ structure.base_only|length if structure.base_only is defined and structure.base_only != none else 0 }}
Number of Variables in Compare Only: {{ structure.compare_only|length if structure.compare_only is defined and structure.compare_only != none else 0 }}

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
Observations in Base:           {{ "%6d"|format(structure.base_nrows if structure.base_nrows is defined and structure.base_nrows != none else 0) }}
Observations in Compare:       {{ "%6d"|format(structure.compare_nrows if structure.compare_nrows is defined and structure.compare_nrows != none else 0) }}
Number of Observations in Common: {{ "%6d"|format(structure.matched_rows if structure.matched_rows is defined and structure.matched_rows != none else 0) }}
Number of Observations in Base Only: {{ "%6d"|format(structure.base_only_rows if structure.base_only_rows is defined and structure.base_only_rows != none else 0) }}
Number of Observations in Compare Only: {{ "%6d"|format(structure.compare_only_rows if structure.compare_only_rows is defined and structure.compare_only_rows != none else 0) }}
                </div>
            </div>

            <div class="section">
                <div class="header">Values Comparison Summary</div>
                <div class="content">
Number of Variables with differences: {{ statistics|length if statistics is defined and statistics != none else 0 }}
Total Number of differences: {{ total_differences if total_differences is defined and total_differences != none else 0 }}
            </div>

            <div class="section">
                <div class="header">Value Comparison Results</div>
                <div class="content">
{% if statistics is defined and statistics != none %}{% for col, stats in statistics.items() %}
Variable: {{ col }}    Type: {{ structure.variable_types.get(col, 'Unknown') }}
    Number of differences: {{ stats.get('n_differences', 0) }}
    {% if stats.max_diff is defined and stats.max_diff != none %}Maximum Absolute Difference: {{ "%.4f"|format(stats.max_diff) }}
    {% endif %}{% if stats.mean_diff is defined and stats.mean_diff != none %}Mean Difference: {{ "%.4f"|format(stats.mean_diff) }}{% endif %}

    First 20 Difference(s) (see CSV export for complete list):
    <div class="diff-container">
        <table class="diff-header">
            <tr>
                <th class="number" style="width: 80px">Obs#</th>
                <th style="width: 160px">Base Value</th>
                <th style="width: 160px">Compare Value</th>
                <th class="number" style="width: 120px">Difference</th>
                <th class="number" style="width: 120px">% Difference</th>
            </tr>
        </table>
        <table class="diff-table">
        {% for diff in stats.get('first_n_differences', []) %}
        <tr>
            <td class="number" style="width: 80px">{{ "%6d"|format(diff.get('obs', 0)) }}</td>
            <td style="width: 160px">{% if diff.base is not defined or diff.base == none %}NULL{% else %}{{ "%.4f"|format(diff.base|float) }}{% endif %}</td>
            <td style="width: 160px">{% if diff.compare is not defined or diff.compare == none %}NULL{% else %}{{ "%.4f"|format(diff.compare|float) }}{% endif %}</td>
            <td class="number" style="width: 120px">{% if diff.abs_diff is defined and diff.abs_diff != none %}{{ "%12.4f"|format(diff.abs_diff|float) }}{% else %}            -{% endif %}</td>
            <td class="number" style="width: 120px">{% if diff.pct_diff is defined and diff.pct_diff != none %}{{ "%12.2f"|format(diff.pct_diff|float) }}%{% else %}            -{% endif %}</td>
        </tr>
        {% endfor %}
        </table>
    </div>
{% endfor %}{% endif %}
                </div>
            </div>
        </body>
        </html>
        """)
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
            for diff in stats.get("first_n_differences", []):  # Note: Now contains all differences
                # Create row data with standard fields
                row_data = {
                    "Variable": str(col),
                    "Observation": str(diff.get("obs", "")),
                    "Base_Value": f"{float(diff.get('base')):.4f}" if diff.get("base") is not None else "NULL",
                    "Compare_Value": f"{float(diff.get('compare')):.4f}" if diff.get("compare") is not None else "NULL",
                    "Difference": f"{float(diff.get('abs_diff')):.4f}" if diff.get("abs_diff") is not None else "",
                    "Pct_Difference": f"{float(diff.get('pct_diff')):.2f}" if diff.get("pct_diff") is not None else ""
                }
                rows.append(row_data)

        # Convert to Polars DataFrame with explicit schema
        if rows:
            # Define schema
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