"""Results handling for DataFrame comparisons."""

from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl
from jinja2 import Template


class ComparisonResults:
    """Handles comparison results and report generation.
    
    This class manages the results of DataFrame comparisons and provides
    functionality to export results in various formats (HTML, CSV).
    
    Attributes:
        structure_results (Dict): Results of structural comparison
        comparison_results (Dict): Results of value comparison
        total_differences (int): Total number of differences found
        optimization_stats (Dict): Memory optimization statistics
    """

    def __init__(self):
        """Initialize comparison results."""
        self.structure_results = {}
        self.comparison_results = {}
        self.total_differences = 0
        self.optimization_stats = {}

    @property
    def statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison statistics.
        
        This is an alias for comparison_results for backward compatibility.
        
        Returns:
            Dictionary mapping column names to comparison statistics:
            - n_differences (int): Number of differences in column
            - first_n_differences (List[Dict]): Sample of differences
            - max_diff (Optional[float]): Maximum difference if numeric
            - mean_diff (Optional[float]): Mean difference if numeric
        """
        return self.comparison_results

    def to_csv(self, output_path: str) -> None:
        """Export differences to CSV.
        
        Creates a CSV file containing all differences found, with columns:
        - Variable: Name of the column
        - Observation: Row identifier
        - Base_Value: Value in base DataFrame
        - Compare_Value: Value in comparison DataFrame
        - Difference: Absolute difference (numeric only)
        - Pct_Difference: Percentage difference (numeric only)
        
        Args:
            output_path: Path where CSV file will be saved
        """
        if not self.statistics:
            return

        # Create a list to store all differences
        rows = []
        for col, stats in self.statistics.items():
            for diff in stats.get("first_n_differences", []):
                # Handle both numeric and non-numeric values
                base_val = diff.get("base")
                comp_val = diff.get("compare")
                abs_diff = diff.get("abs_diff")
                pct_diff = diff.get("pct_diff")

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
                    "Pct_Difference": pct_str,
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
                ("Pct_Difference", pl.Utf8),
            ]

            df = pl.DataFrame(rows, schema=schema)
            df.write_csv(output_path)

    def set_structure_results(self, results: Dict[str, Any]) -> None:
        """Set structure comparison results."""
        self.structure_results = results

    def set_comparison_results(self, stats: Dict[str, Dict[str, Any]], total_differences: int) -> None:
        """Set value comparison results."""
        self.comparison_results = stats
        self.total_differences = total_differences

    def _generate_html(self) -> str:
        """Generate HTML report content.
        
        Returns:
            str: The generated HTML content
        """
        if not self.structure_results:
            return ""

        template = Template(
            """
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Comparison Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }

        /* Summary Section */
        .summary {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .summary h2 {
            color: #2c3e50;
            margin-top: 0;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }

        .summary h3 {
            color: #2c3e50;
            margin-top: 20px;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Table Headers */
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
            padding: 12px;
            text-align: left;
            border: 1px solid #2980b9;
        }

        /* Table Cells */
        td {
            padding: 10px;
            border: 1px solid #dee2e6;
        }

        /* Numeric Columns */
        td.numeric {
            text-align: right;
            font-family: 'Courier New', monospace;
        }

        /* String Columns */
        td.text {
            text-align: left;
        }

        /* Alternating Rows */
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        tr:nth-child(odd) {
            background-color: white;
        }

        /* Row Hover Effect */
        tr:hover {
            background-color: #edf2f7;
        }

        /* Section Headers */
        .section-header {
            background-color: #2c3e50;
            color: white;
            padding: 10px 15px;
            margin: 30px 0 15px 0;
            border-radius: 4px;
        }

        /* Statistics Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            text-align: center;
        }

        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }

        .stat-label {
            color: #666;
            font-size: 14px;
        }

        /* Difference Highlighting */
        .difference {
            color: #e74c3c;
            font-weight: bold;
        }

        /* Dataset Summary Table */
        .dataset-summary {
            margin: 20px 0;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            overflow: hidden;
        }

        .dataset-summary table {
            margin: 0;
        }

        .dataset-summary h3 {
            background-color: #3498db;
            color: white;
            padding: 10px 15px;
            margin: 0;
        }

        .dataset-info {
            display: grid;
            grid-template-columns: auto auto auto;
            gap: 1px;
            background-color: #dee2e6;
        }

        .dataset-info-cell {
            background-color: white;
            padding: 12px 15px;
        }

        .dataset-info-header {
            background-color: #f8f9fa;
            font-weight: bold;
            text-align: center;
        }

        .dataset-row-header {
            background-color: #f8f9fa;
            font-weight: bold;
        }

        /* Summary Sections */
        .summary-section {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            margin: 20px 0;
            overflow: hidden;
        }

        .summary-section h3 {
            background-color: #3498db;
            color: white;
            padding: 12px 15px;
            margin: 0;
            font-size: 16px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1px;
            background-color: #dee2e6;
            padding: 1px;
        }

        .info-cell {
            background-color: white;
            padding: 15px;
            text-align: center;
        }

        .info-value {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }

        .info-label {
            color: #666;
            font-size: 13px;
        }

        .info-cell.highlight .info-value {
            color: #e74c3c;
        }

        /* Column Difference Sections */
        .column-diff-section {
            cursor: pointer;
            scroll-margin-top: 20px;
            margin-bottom: 20px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            overflow: hidden;
        }

        .column-header {
            background: #3498db;
            color: white;
            padding: 12px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 20px;
            z-index: 1;
        }

        .column-content {
            display: none;
            background: white;
            padding: 15px;
            transition: all 0.3s ease-in-out;
        }

        .column-diff-section.active .column-content {
            display: block;
        }

        .column-stats {
            background: #f8f9fa;
            padding: 15px;
            display: flex;
            gap: 25px;
            border-bottom: 1px solid #dee2e6;
            margin-top: 1px;
        }

        .stat-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 5px 0;
        }

        .stat-item .stat-label {
            color: #666;
            font-size: 13px;
            font-weight: 500;
        }

        .stat-item .stat-value {
            font-weight: bold;
            color: #2c3e50;
            font-size: 16px;
        }

        .diff-table {
            margin: 0 !important;
            box-shadow: none !important;
        }

        .diff-table th {
            background-color: #f8f9fa;
            color: #2c3e50;
            font-weight: bold;
            text-align: right;
        }

        .diff-table th:first-child {
            text-align: left;
        }

        .diff-table td.numeric {
            font-family: 'Courier New', monospace;
            text-align: right;
            padding: 8px 12px;
        }

        .diff-table td:first-child {
            text-align: left;
            font-weight: bold;
        }

        .diff-table tr:hover {
            background-color: #f1f7fe;
        }

        /* More Differences Row */
        .more-differences td {
            text-align: center !important;
            color: #666;
            font-style: italic;
            background-color: #f8f9fa;
            padding: 8px !important;
        }

        /* Comparison Results Section */
        .comparison-results {
            margin-top: 30px;
        }

        .comparison-results h2 {
            background-color: #2c3e50;
            color: white;
            padding: 12px 15px;
            margin: 0 0 20px 0;
            border-radius: 4px 4px 0 0;
        }
    </style>
    <script>
        (function() {
            function setupAccordions() {
                const sections = document.querySelectorAll('.column-diff-section');
                
                // Function to close all sections except the one being opened
                function closeOtherSections(currentSection) {
                    sections.forEach(section => {
                        if (section !== currentSection) {
                            section.classList.remove('active');
                        }
                    });
                }
                
                // Set up click handlers
                sections.forEach(section => {
                    const header = section.querySelector('.column-header');
                    if (header) {
                        header.onclick = function() {
                            const wasActive = section.classList.contains('active');
                            closeOtherSections(section);
                            
                            // Toggle current section
                            if (!wasActive) {
                                section.classList.add('active');
                                // Smooth scroll into view
                                setTimeout(() => {
                                    section.scrollIntoView({ 
                                        behavior: 'smooth',
                                        block: 'start'
                                    });
                                }, 10);
                            }
                        };
                    }
                });
                
                // Open first section by default
                if (sections.length > 0) {
                    sections[0].classList.add('active');
                }
            }

            // For regular browser context
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', setupAccordions);
            } else {
                setupAccordions();
            }

            // For Jupyter notebook context
            setTimeout(setupAccordions, 100);
            
            // Additional call after a longer delay to ensure it works in all contexts
            setTimeout(setupAccordions, 500);
        })();
    </script>
</head>
<body>
    <div class="summary">
        <h2>Dataset Comparison Report</h2>
        
        <div class="dataset-summary">
            <h3>Dataset Summary</h3>
            <div class="dataset-info">
                <div class="dataset-info-cell dataset-info-header"></div>
                <div class="dataset-info-cell dataset-info-header">Observations</div>
                <div class="dataset-info-cell dataset-info-header">Variables</div>
                
                <div class="dataset-info-cell dataset-row-header">Base</div>
                <div class="dataset-info-cell">{{base_nrows}}</div>
                <div class="dataset-info-cell">{{base_ncols}}</div>
                
                <div class="dataset-info-cell dataset-row-header">Compare</div>
                <div class="dataset-info-cell">{{compare_nrows}}</div>
                <div class="dataset-info-cell">{{compare_ncols}}</div>
            </div>
        </div>

        <div class="summary-section">
            <h3>Variables Summary</h3>
            <div class="info-grid">
                <div class="info-cell">
                    <div class="info-value">{{n_common}}</div>
                    <div class="info-label">Variables in Common</div>
                </div>
                <div class="info-cell">
                    <div class="info-value">{{n_base_only}}</div>
                    <div class="info-label">Variables in Base Only</div>
                </div>
                <div class="info-cell">
                    <div class="info-value">{{n_compare_only}}</div>
                    <div class="info-label">Variables in Compare Only</div>
                </div>
            </div>
        </div>

        <div class="summary-section">
            <h3>Values Comparison Summary</h3>
            <div class="info-grid">
                <div class="info-cell highlight">
                    <div class="info-value">{{total_differences}}</div>
                    <div class="info-label">Differences Found</div>
                </div>
                <div class="info-cell highlight">
                    <div class="info-value">{{n_vars_with_diffs}}</div>
                    <div class="info-label">Columns with Differences</div>
                </div>
            </div>
        </div>

        {% if optimization_stats %}
        <div class="summary-section">
            <h3>Memory Optimization</h3>
            <div class="info-grid">
                <div class="info-cell">
                    <div class="info-value">{{ (optimization_stats.original_size / (1024 * 1024))|round(2) }}</div>
                    <div class="info-label">Original Size (MB)</div>
                </div>
                <div class="info-cell">
                    <div class="info-value">{{ (optimization_stats.optimized_size / (1024 * 1024))|round(2) }}</div>
                    <div class="info-label">Optimized Size (MB)</div>
                </div>
                <div class="info-cell">
                    <div class="info-value">{{ optimization_stats.reduction_percent|round(1) }}%</div>
                    <div class="info-label">Memory Reduction</div>
                </div>
            </div>
        </div>
        {% endif %}
    </div>

    <div class="comparison-results">
        <h2>Detailed Comparison Results</h2>
        {% for var, stats in statistics.items() %}
        <div class="column-diff-section">
            <div class="column-header">
                <div class="column-name">{{var}}</div>
                <div class="column-type">Type: {{variable_types[var]}}</div>
            </div>
            <div class="column-content">
                <div class="column-stats">
                    <div class="stat-item">
                        <span class="stat-label">Differences:</span>
                        <span class="stat-value">{{stats.n_differences}}</span>
                    </div>
                    {% if stats.max_diff is not none %}
                    <div class="stat-item">
                        <span class="stat-label">Max Difference:</span>
                        <span class="stat-value">{{stats.max_diff}}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Mean Difference:</span>
                        <span class="stat-value">{{stats.mean_diff}}</span>
                    </div>
                    {% endif %}
                </div>
                <table class="diff-table">
                    <thead>
                        <tr>
                            <th>Observation</th>
                            <th>Base Value</th>
                            <th>Compare Value</th>
                            <th>Difference</th>
                            <th>% Difference</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for diff in stats.first_n_differences %}
                        <tr>
                            <td>{{diff.obs}}</td>
                            <td class="numeric">{{diff.base}}</td>
                            <td class="numeric">{{diff.compare}}</td>
                            <td class="numeric">{{diff.abs_diff}}</td>
                            <td class="numeric">{{diff.pct_diff}}</td>
                        </tr>
                    {% endfor %}
                    {% if stats.n_differences > stats.first_n_differences|length %}
                        <tr class="more-differences">
                            <td colspan="5">... and {{stats.n_differences - stats.first_n_differences|length}} more differences</td>
                        </tr>
                    {% endif %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
            """
        )

        return template.render(
            base_nrows=self.structure_results["base_nrows"],
            compare_nrows=self.structure_results["compare_nrows"],
            base_ncols=self.structure_results["base_ncols"],
            compare_ncols=self.structure_results["compare_ncols"],
            n_common=len(self.structure_results["common_cols"]),
            n_base_only=len(self.structure_results["base_only"]),
            n_compare_only=len(self.structure_results["compare_only"]),
            matched_rows=self.structure_results["matched_rows"],
            base_only_rows=self.structure_results["base_only_rows"],
            compare_only_rows=self.structure_results["compare_only_rows"],
            n_vars_with_diffs=len(self.comparison_results),
            total_differences=self.total_differences,
            variable_types=self.structure_results["variable_types"],
            statistics=self.statistics,
            optimization_stats=self.optimization_stats
        )

    def to_html(self, output_path: str) -> None:
        """Generate and save HTML report.
        
        Creates a detailed HTML report containing:
        - Dataset summary (rows, columns)
        - Variable comparison summary
        - Observation summary
        - Value comparison details
        - Memory optimization statistics
        
        Args:
            output_path: Path where HTML report will be saved
        """
        html_content = self._generate_html()
        if html_content:
            Path(output_path).write_text(html_content)

    def display_html(self) -> None:
        """Display comparison results as HTML in the notebook.
        
        This method renders the HTML report directly in the Jupyter notebook.
        It uses the same template as to_html() but displays inline instead of
        saving to a file.
        """
        try:
            from IPython.display import HTML, display
            html_content = self._generate_html()
            if html_content:
                display(HTML(html_content))
        except ImportError:
            print("IPython display functionality not available. "
                  "Use to_html() to save the report to a file instead.")
