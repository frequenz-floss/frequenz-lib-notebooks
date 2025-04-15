# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
This module provides functionality for generating email alert notifications.

It includes functions for formatting and structuring alert-related emails,
such as:
    - Generating a summary of alerts per microgrid (optionally grouped by
      component ID).
    - Creating an HTML table representation of alert details.
    - Constructing a complete alert email with formatted content.
    - Sorting alerts by severity (optional) and applying color-coded styling.
    - Generating structured JSON output for alerts.
    - Filtering groups with no errors or warnings (optional, enabled by default).

### Example Usage:
```python
import pandas as pd
from frequenz.lib.notebooks.alerts.alert_email import AlertEmailConfig, generate_alert_email

def example():
    # Example alert records dataframe
    alert_records = pd.DataFrame(
        [
            {
                "microgrid_id": 1,
                "component_id": 1,
                "state_type": "error",
                "state_value": "UNDERVOLTAGE",
                "start_time": "2025-03-14 15:06:30",
                "end_time": "2025-03-14 17:00:00",
            },
            {
                "microgrid_id": 2,
                "component_id": 1,
                "state_type": "state",
                "state_value": "DISCHARGING",
                "start_time": "2025-03-14 15:06:30",
                "end_time": None,
            },
        ]
    )

    # Configuration for email generation
    alert_email_config = AlertEmailConfig(
        notebook_url="http://alerts.example.com",
        displayed_rows=10,
        sort_by_severity=True,
        group_by_component=False,
        filter_no_alerts=True,
    )

    # Generate the HTML body of the alert email
    html_email = generate_alert_email(alert_records=alert_records, config=alert_email_config)

    # Output the HTML or send it via email
    print(html_email)
```
"""
import html
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.io as pio
from pandas import Series
from pandas.api.types import is_scalar

_log = logging.getLogger(__name__)

EMAIL_CSS = """
<style>
    body { font-family: 'Roboto', sans-serif; line-height: 1.6; }
    .table { width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }
    .table th, .table td { border: 1px solid #ddd; padding: 8px; }
    .table th { background-color: #f4f4f4; font-weight: bold; }
</style>
"""

SEVERITY_ORDER = ["error", "warning", "state"]
T = TypeVar("T", bound=Enum)

if TYPE_CHECKING:
    SeriesType = Series[Any]
else:
    SeriesType = Series


class AlertPlotType(str, Enum):
    """Possible plot types for alert visualisations."""

    SUMMARY = "summary"
    """Plot of alert counts per microgrid/component."""

    STATE_TRANSITIONS = "state_transitions"
    """Plot of state transitions over time."""

    ALL = "all"
    """Generate all available plots."""


class ImageExportFormat(str, Enum):
    """Export formats for images."""

    PNG = "png"
    HTML = "html"
    SVG = "svg"
    PDF = "pdf"
    JPEG = "jpeg"
    JSON = "json"


@dataclass
class ExportOptions:
    """Configuration for exporting and/or displaying plots."""

    format: str | ImageExportFormat | list[str | ImageExportFormat] | None = field(
        default=None,
        metadata={
            "description": (
                "Export format(s) for the plots. Examples: 'png', ['html', 'svg']. "
                "All options: 'png', 'html', 'svg', 'pdf', 'jpeg', 'json'. "
                "If None, plots will not be saved."
            ),
        },
    )

    output_dir: str | Path | None = field(
        default=None,
        metadata={
            "description": (
                "Directory to save the exported plots. "
                "If None, uses the current directory."
            ),
        },
    )

    show: bool = field(
        default=True,
        metadata={"description": "Whether to display the plots interactively."},
    )


@dataclass(kw_only=True)
class AlertEmailConfig:
    """Configuration for generating alert emails."""

    notebook_url: str = field(
        default="",
        metadata={
            "description": "URL to manage alert preferences.",
        },
    )

    displayed_rows: int = field(
        default=20,
        metadata={
            "description": "Number of alert rows to display in the HTML table.",
            "validate": lambda x: x > 0,
        },
    )

    sort_by_severity: bool = field(
        default=False,
        metadata={
            "description": "Whether to sort alerts by severity level in the HTML table.",
        },
    )

    group_by_component: bool = field(
        default=False,
        metadata={
            "description": (
                "Whether to group summary by component_id in addition to "
                "microgrid_id in the HTML table."
            ),
        },
    )

    filter_no_alerts: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether to exclude groups with no errors or warnings "
                "in the alert email."
            ),
        },
    )


def compute_time_since(row: SeriesType, ts_column: str) -> str:
    """Calculate the time elapsed since a given timestamp (start or end time).

    Args:
        row: DataFrame row containing timestamps.
        ts_column: Column name ("start_time" or "end_time") to compute from.

    Returns:
        Time elapsed as a formatted string (e.g., "3h 47m", "2d 5h").
    """
    timestamp = _parse_and_localize_timestamp(row[ts_column])
    now = pd.Timestamp.utcnow()

    if pd.isna(timestamp):
        return "N/A"

    if ts_column == "start_time":
        end_time = _parse_and_localize_timestamp(row["end_time"])
        reference_time = end_time if pd.notna(end_time) else now
    else:
        reference_time = now

    return _format_timedelta(reference_time - timestamp)


def _format_timedelta(delta: timedelta) -> str:
    """Format a timedelta object into a human-readable string.

    Args:
        delta: Timedelta object representing time difference.

    Returns:
        Formatted string (e.g., "3h 47m", "2d 5h"). Defaults to "0s" if zero.
    """
    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Build output dynamically
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 and not days:
        parts.append(f"{seconds}s")

    return " ".join(parts) if parts else "0s"


def _parse_and_localize_timestamp(timestamp: Any) -> pd.Timestamp | None:
    """Parse a timestamp, coerce errors to NaT, and localize to UTC if naive.

    Args:
        timestamp: The timestamp value to process.

    Returns:
        A timezone-aware Pandas Timestamp, or None if the input is not a scalar.
    """
    if not is_scalar(timestamp):
        return None
    parsed_time = pd.to_datetime(timestamp, errors="coerce")
    if pd.notna(parsed_time) and parsed_time.tz is None:
        return pd.Timestamp(parsed_time.tz_localize("UTC"))
    return pd.Timestamp(parsed_time)


def generate_alert_summary(
    alert_records: pd.DataFrame,
    group_by_component: bool = False,
    filter_no_alerts: bool = True,
) -> str:
    """Generate a summary of alerts per microgrid, optionally grouped by component ID.

    Args:
        alert_records: DataFrame containing alert records.
        group_by_component: Whether to group alerts by component ID.
        filter_no_alerts: Whether to exclude groups with zero errors and warnings.

    Returns:
        HTML summary string.
    """
    if alert_records.empty:
        return "<p>No alerts recorded.</p>"

    group_columns = ["microgrid_id"]
    if group_by_component:
        group_columns.append("component_id")

    summary_data = (
        alert_records.groupby(group_columns)
        .agg(
            total_errors=(
                "state_type",
                lambda x: (x.fillna("").str.lower() == "error").sum(),
            ),
            total_warnings=(
                "state_type",
                lambda x: (x.fillna("").str.lower() == "warning").sum(),
            ),
            unique_states=(
                "state_value",
                lambda x: [html.escape(str(s)) for s in x.unique()],
            ),
            unique_components=("component_id", lambda x: list(x.unique())),
        )
        .reset_index()
    )

    if filter_no_alerts:
        summary_data = summary_data[
            (summary_data["total_errors"] > 0) | (summary_data["total_warnings"] > 0)
        ]

    summary_html = "".join(
        [
            f"""
            <p><strong>Microgrid {row['microgrid_id']}{
                ", Component " + str(row['component_id']) if group_by_component else ""
                }:</strong></p>
            <ul>
                <li><strong>Total errors:</strong> {row['total_errors']}</li>
                <li><strong>Total warnings:</strong> {row['total_warnings']}</li>
                <li><strong>States:</strong>
                    <ul>
                        <li>Unique states found: {len(row['unique_states'])}</li>
                        <li>Unique States: {row['unique_states']}</li>
                    </ul>
                </li>
            </ul>
            """
            + (
                f"""
                <ul>
                    <li><strong>Components:</strong>
                        <ul>
                            <li>Alerts found for {len(row['unique_components'])} components</li>
                            <li>Components: {row['unique_components']}</li>
                        </ul>
                    </li>
                </ul>
                """
                if not group_by_component
                else ""
            )
            + "</p>"
            for _, row in summary_data.iterrows()
        ]
    )

    return summary_html


def generate_alert_table(
    alert_records: pd.DataFrame,
    displayed_rows: int = 20,
    sort_by_severity: bool = False,
) -> str:
    """Generate a formatted HTML table for alert details with color-coded severity levels.

    Args:
        alert_records: DataFrame containing alert records.
        displayed_rows: Number of rows to display.
        sort_by_severity: Whether to sort alerts by severity.

    Returns:
        HTML string of the table with color-coded rows.
    """
    if alert_records.empty:
        return "<p>No alerts recorded.</p>"

    if sort_by_severity:
        alert_records = alert_records.copy()
        alert_records["state_type"] = alert_records["state_type"].str.lower()
        alert_records["state_type"] = pd.Categorical(
            alert_records["state_type"], categories=SEVERITY_ORDER, ordered=True
        )
        alert_records = alert_records.sort_values("state_type")

    if len(alert_records) > displayed_rows:
        note = f"""
        <p><strong>Note:</strong> Table limited to {displayed_rows} rows.
        Download the attached file to view all {len(alert_records)} rows.</p>
        """
    else:
        note = ""

    severity_colors = {
        "error": "background-color: #D32F2F; color: white;",
        "warning": "background-color: #F57C00; color: black;",
    }

    # general table styling
    # We use cast(Any, ...) here to bypass mypy's strict type checking on .set_table_styles().
    # Pandas internally expects CSSDict, which we want to avoid importing to prevent
    # unnecessary dependencies (like with Jinja2 library).
    # Since the structure is guaranteed to be correct for Pandas at runtime, using Any
    # ensures type flexibility while avoiding compatibility issues with future Pandas versions.
    table_styles = cast(
        Any,
        [
            {
                "selector": "th",
                "props": [("background-color", "#f4f4f4"), ("font-weight", "bold")],
            },
            {
                "selector": "td, th",
                "props": [("border", "1px solid #ddd"), ("padding", "8px")],
            },
        ],
    )

    # apply severity color to entire rows
    styled_table = (
        alert_records.head(displayed_rows)
        .style.apply(
            lambda row: [severity_colors.get(row["state_type"], "")] * len(row), axis=1
        )
        .set_table_styles(table_styles, overwrite=False)
        .hide(axis="index")
        .to_html()
    )
    return f"{note}{styled_table}"


def generate_alert_json(
    alert_records: pd.DataFrame, group_by_component: bool = False
) -> dict[str, Any]:
    """Generate a JSON representation of the alert data.

    The data can be optionally grouped by component ID

    Args:
        alert_records: DataFrame containing alert records.
        group_by_component: Whether to group alerts by component ID.

    Returns:
        Dictionary representing the alert data in JSON format.
    """
    if alert_records.empty:
        return {"summary": "<p>No alerts recorded.</p>"}

    group_columns = ["microgrid_id"]
    if group_by_component:
        group_columns.append("component_id")

    return {
        "summary": {
            idx: group.to_dict(orient="records")
            for idx, group in alert_records.groupby(group_columns)
        }
    }


def generate_alert_email(
    alert_records: pd.DataFrame,
    config: AlertEmailConfig,
    checks: list[str] | None = None,
) -> str:
    """Generate a full HTML email for alerts.

    Args:
        alert_records: DataFrame containing alert records.
        config: Configuration object for email generation.
        checks: A list of conditions checked by the alert system.

    Returns:
        Full HTML email body.
    """
    return f"""
    <html>
        <head>{EMAIL_CSS}</head>
        <body>
            <h1>Microgrid Alert</h1>
            <h2>Summary:</h2>
            {generate_alert_summary(alert_records, config.group_by_component,
                                    config.filter_no_alerts)}
            <h2>Alert Details:</h2>
            {generate_alert_table(alert_records, config.displayed_rows, config.sort_by_severity)}
            {generate_check_status(checks)}
            <p style='color: #665;'><em>This is an automated notification.</em></p>
            {_generate_email_footer(config.notebook_url)}
        </body>
    </html>
    """


def generate_check_status(checks: list[str] | None) -> str:
    """Generate a clean HTML bullet list summarising what was checked.

    Args:
        checks: A list of plain text items (e.g. conditions, rules).

    Returns:
        An HTML unordered list.
    """
    if not checks:
        return ""
    items = "".join(f"<li>{html.escape(check)}</li>" for check in checks)
    return (
        "<h3>✅ Conditions Checked:</h3>"
        "<ul style='line-height: 1.6;'>"
        f"{items}"
        "</ul>"
    )


def generate_no_alerts_email(
    checks: list[str] | None = None,
    notebook_url: str = "",
) -> str:
    """Generate an HTML email when no alerts are found.

    Args:
        checks: A list of conditions checked.
        notebook_url: Optional link to manage preferences.

    Returns:
        HTML email body.
    """
    return f"""
    <html>
    <body style='font-family: Arial, sans-serif; line-height: 1.6; color: #333;'>
        <h2 style='color: #28a745;'>✅ No Alerts Detected</h2>
        <p>
        This is a confirmation that the alerting system ran successfully and no
        alerts were found at this time.
        </p>
        {generate_check_status(checks)}
        <p style='color: #665;'><em>This is an automated notification.</em></p>
        {_generate_email_footer(notebook_url)}
    </body>
    </html>
    """


def _generate_email_footer(notebook_url: str = "") -> str:
    """Generate a shared footer for email messages.

    Args:
        notebook_url: Optional URL for managing alert preferences.

    Returns:
        HTML footer string.
    """
    footer = (
        f"<p><a href='{html.escape(notebook_url)}'>Manage Alert Preferences</a></p>"
        if notebook_url
        else ""
    )
    return f"""
    <hr>
    <div class="footer" style="text-align: center; font-size: 12px; color: #777;">
        <p>&copy; 2024 Frequenz Energy-as-a-Service GmbH. All rights reserved.</p>
        {footer}
    </div>
    """


def plot_alerts(
    records: pd.DataFrame,
    *,
    plot_type: str | AlertPlotType = AlertPlotType.SUMMARY,
    export_options: ExportOptions | None = None,
    **kwargs: Any,
) -> list[str] | None:
    """Generate alert visualisations and optionally save them as image files.

    Behaviour based on `export_options` `format` and `show` fields:
        - format=None, show=False: Do nothing.
        - format=None, show=True: Display plots only (default).
        - format=[...], show=False: Save plots to multiple formats only.
        - format=[...], show=True: Save plots to multiple formats and display them.

    Args:
        records: DataFrame containing alert records with expected columns
            "microgrid_id", "component_id", "state_value", and "start_time".
        plot_type: Which plot to create. Options:
            - 'summary': Plot of alert counts per microgrid/component.
            - 'state_transitions': Plot of state transitions over time.
            - 'all': Generate both types.
        export_options: Configuration for exporting and/or displaying the plots.
        **kwargs: Additional arguments for the plot functions.
            - 'stacked': Whether to use a stacked bar representation per microgrid
                (only for 'summary' plot type).

    Returns:
        List of file paths if plots are exported, otherwise None.
    """
    if records.empty:
        _log.info("Records are empty, no plots generated.")
        return None

    if (
        export_options is not None
        and export_options.format is None
        and not export_options.show
    ):
        _log.info("No export format and show=False: no plots generated.")
        return None

    plot_type = _coerce_enum(plot_type, AlertPlotType, "plot_type")
    figs = {}
    if plot_type in (AlertPlotType.SUMMARY, AlertPlotType.ALL):
        figs["alert_summary.html"] = _plot_alert_summary(records, **kwargs)
    if plot_type in (AlertPlotType.STATE_TRANSITIONS, AlertPlotType.ALL):
        figs["state_transitions.html"] = _plot_state_transitions(records)

    filepaths = None
    if export_options is not None:
        if export_options.format is not None:
            export_formats = _coerce_formats(export_options.format)
            filepaths = []
            for fmt in export_formats:
                filepaths += _save_figures(figs, fmt, export_options.output_dir)

        if export_options.show:
            for fig in figs.values():
                fig.show()
    return filepaths


def _save_figures(
    figs: dict[str, go.Figure],
    export_format: ImageExportFormat,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Save Plotly figures to files in the specified format.

    Args:
        figs: Dictionary of figure names and Plotly figures.
        export_format: The format to export the figures.
        output_dir: Directory to save the files. If the directory does not exist,
            it will be created. If None, uses the current directory.

    Returns:
        A list of file paths where the figures were saved.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    filepaths: list[str] = []
    for name, fig in figs.items():
        filename = f"{name}.{export_format.value}"
        file_path = os.path.join(output_dir, filename)

        if export_format == ImageExportFormat.HTML:
            pio.write_html(fig, file=file_path, include_plotlyjs="cdn", full_html=False)
        elif export_format == ImageExportFormat.JSON:
            pio.write_json(fig, file=file_path)
        else:
            pio.write_image(fig, file=file_path, format=export_format.value)

        _log.info("Plot saved to %s", file_path)
        filepaths.append(file_path)
    return filepaths


def _coerce_formats(
    formats: str | ImageExportFormat | list[str | ImageExportFormat],
) -> list[ImageExportFormat]:
    """Coerce a format or list of formats to a list of ImageExportFormat enums.

    Args:
        formats: A format string, an ImageExportFormat enum, or a list of either.

    Returns:
        A list of ImageExportFormat enums.
    """
    if isinstance(formats, (str, Enum)):
        formats = [formats]
    return [_coerce_enum(fmt, ImageExportFormat, "format") for fmt in formats]


def _coerce_enum(value: str | T, enum_type: type[T], param_name: str) -> T:
    """Coerce a value to an enum type.

    Args:
        value: The value to coerce.
        enum_type: The enum type to coerce to.
        param_name: The name of the parameter for error messages.

    Returns:
        The coerced enum value.

    Raises:
        ValueError: If the value is not a valid enum member or string.
        TypeError: If the value is not a string or enum type.
    """
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value.lower())
        except ValueError as er:
            valid = [e.value for e in enum_type]
            raise ValueError(
                f"Invalid {param_name} '{value}'. Expected one of: {valid}"
            ) from er
    raise TypeError(
        f"{param_name!r} must be a string or {enum_type.__name__}, got {type(value).__name__}"
    )


def _plot_alert_summary(df: pd.DataFrame, stacked: bool = True) -> go.Figure:
    """Plot alert summary by microgrid and component.

    Args:
        df: DataFrame containing alert records with these expected columns:
            "microgrid_id" and "component_id".
        stacked: Whether to use a stacked bar representation per microgrid.

    Returns:
        A Plotly figure.
    """

    def norm(x: float, bounds: tuple[float, float]) -> float:
        """Normalize x to a value between 0 and 1.

        Args:
            x: The value to normalize.
            bounds: A tuple containing the minimum and maximum values for
                normalization.

        Returns:
            Normalized value between 0 and 1.
        """
        min_count, max_count = bounds
        return (
            (x - min_count) / (max_count - min_count) if max_count != min_count else 0.5
        )

    grouped = (
        df.groupby(["microgrid_id", "component_id"])
        .size()
        .reset_index(name="alert_count")
    ).sort_values(by=["microgrid_id", "alert_count"], ascending=[True, False])

    labels = grouped.apply(
        lambda row: f"{row['microgrid_id']}.{row['component_id']}", axis=1
    )

    fig = go.Figure()
    if not stacked:
        fig.add_trace(
            go.Bar(x=labels, y=grouped["alert_count"], marker_color="#607d8b")
        )
    else:
        for i, microgrid_id in enumerate(grouped["microgrid_id"].unique()):
            blues_r = pc.sequential.Blues
            microgrid_data = grouped[grouped["microgrid_id"] == microgrid_id]
            alert_counts = microgrid_data["alert_count"].values
            min_count, max_count = alert_counts.min(), alert_counts.max()

            fig.add_trace(
                go.Bar(
                    x=[str(microgrid_id)] * len(microgrid_data),
                    y=alert_counts,
                    name=f"Microgrid {microgrid_id}",
                    text=[
                        f"CID {comp_id}: {count} alerts"
                        for comp_id, count in zip(
                            microgrid_data["component_id"], alert_counts
                        )
                    ],
                    customdata=microgrid_data["component_id"],
                    hovertemplate=(
                        "Microgrid: %{x}<br>Component: %{customdata}<br>Alerts: %{y}<extra></extra>"
                    ),
                    textposition="inside" if stacked else "auto",
                    marker_color=[
                        pc.sample_colorscale(blues_r, norm(v, (min_count, max_count)))[
                            0
                        ]
                        for v in alert_counts
                    ],
                )
            )

    fig.update_layout(
        barmode="stack" if stacked else "group",
        title="Alert Breakdown by Component",
        xaxis_title="Microgrid ID" if stacked else "Microgrid ID.Component ID",
        yaxis_title="Number of Alerts",
    )
    return fig


def _plot_state_transitions(df: pd.DataFrame) -> go.Figure:
    """Plot state transitions over time for each microgrid and component.

    Args:
        df: DataFrame containing alert records with these expected columns:
            "microgrid_id", "component_id", "state_value", and "start_time".

    Returns:
        A Plotly figure.
    """
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    # Extend the last state for each microgrid.component to now
    last_rows = (
        df.sort_values("start_time")
        .groupby(["microgrid_id", "component_id"], as_index=False)
        .tail(1)
        .copy()
    )
    last_rows["start_time"] = pd.Timestamp.now(tz="UTC")
    df = pd.concat([df, last_rows], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["start_time"], utc=True)
    df = df.dropna(subset=["timestamp"])

    # Sort states by frequency for improved readability
    state_order = (
        df["state_value"]
        .dropna()
        .value_counts()
        .sort_values(ascending=False)
        .index.tolist()
    )
    state_map = {state: i for i, state in enumerate(state_order)}

    fig = go.Figure()
    for (mg_id, comp_id), group in df.groupby(["microgrid_id", "component_id"]):
        fig.add_trace(
            go.Scatter(
                x=group["timestamp"],
                y=group["state_value"].map(state_map),
                mode="lines+markers",
                name=f"{mg_id}.{comp_id}",
            )
        )
    fig.update_layout(
        title="Alert State Transitions Over Time",
        xaxis_title="Time",
        yaxis_title="State",
        yaxis={
            "tickmode": "array",
            "tickvals": list(state_map.values()),
            "ticktext": list(state_map.keys()),
        },
    )
    return fig
