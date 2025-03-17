# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""This module provides functionality for generating email alert notifications."""
import html
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas import Series

EMAIL_CSS = """
<style>
    body { font-family: 'Roboto', sans-serif; line-height: 1.6; }
    .table { width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }
    .table th, .table td { border: 1px solid #ddd; padding: 8px; }
    .table th { background-color: #f4f4f4; font-weight: bold; }
</style>
"""

if TYPE_CHECKING:
    SeriesType = Series[Any]
else:
    SeriesType = Series


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


def _parse_and_localize_timestamp(timestamp: Any) -> pd.Timestamp:
    """Parse a timestamp, coerce errors to NaT, and localize to UTC if naive.

    Args:
        timestamp: The timestamp value to process.

    Returns:
        A timezone-aware Pandas Timestamp, or NaT if parsing fails.
    """
    parsed_time = pd.to_datetime(timestamp, errors="coerce")
    if pd.notna(parsed_time) and parsed_time.tz is None:
        return pd.Timestamp(parsed_time.tz_localize("UTC"))
    return pd.Timestamp(parsed_time)


def generate_alert_summary(alert_records: pd.DataFrame) -> str:
    """Generate a summary of alerts per microgrid.

    Args:
        alert_records: DataFrame containing alert records.

    Returns:
        HTML summary string.
    """
    if alert_records.empty:
        return "<p>No alerts recorded.</p>"

    summary_data = alert_records.groupby("microgrid_id").agg(
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

    summary_html = "".join(
        [
            f"""
        <p><strong>Microgrid {microgrid_id}:</strong></p>
        <ul>
            <li><strong>Total errors:</strong> {row['total_errors']}</li>
            <li><strong>Total warnings:</strong> {row['total_warnings']}</li>
            <li><strong>States:</strong>
                <ul>
                    <li>Unique states found: {len(row['unique_states'])}</li>
                    <li>Unique States: {row['unique_states']}</li>
                </ul>
            </li>
            <li><strong>Components:</strong>
                <ul>
                    <li>Alerts found for {len(row['unique_components'])} components</li>
                    <li>Components: {row['unique_components']}</li>
                </ul>
            </li>
        </ul>
        """
            for microgrid_id, row in summary_data.iterrows()
        ]
    )
    return summary_html


def generate_alert_table(
    alert_records: pd.DataFrame,
    displayed_rows: int = 20,
) -> str:
    """Generate a formatted HTML table for alert details.

    Args:
        alert_records: DataFrame containing alert records.
        displayed_rows: Number of rows to display.

    Returns:
        HTML string of the table.
    """
    if alert_records.empty:
        return "<p>No alerts recorded.</p>"

    if len(alert_records) > displayed_rows:
        note = f"""
        <p><strong>Note:</strong> Table limited to {displayed_rows} rows.
        Download the attached file to view all {len(alert_records)} rows.</p>
        """
    else:
        note = ""

    styled_table = alert_records.head(displayed_rows).to_html(
        index=False, classes="table table-bordered"
    )
    return f"{note}{styled_table}"


def generate_alert_json(alert_records: pd.DataFrame) -> dict[str, Any]:
    """Generate a JSON representation of the alert data.

    Args:
        alert_records: DataFrame containing alert records.

    Returns:
        Dictionary representing the alert data in JSON format.
    """
    if alert_records.empty:
        return {"summary": "<p>No alerts recorded.</p>"}
    return {
        "summary": {
            microgrid_id: group.to_dict(orient="records")
            for microgrid_id, group in alert_records.groupby("microgrid_id")
        }
    }


def generate_alert_email(
    alert_records: pd.DataFrame,
    notebook_url: str,
    displayed_rows: int = 20,
) -> str:
    """Generate a full HTML email for alerts.

    Args:
        alert_records: DataFrame containing alert records.
        notebook_url: URL for managing alert preferences.
        displayed_rows: Number of rows to display in the email.

    Returns:
        Full HTML email body.
    """
    return f"""
    <html>
        <head>{EMAIL_CSS}</head>
        <body>
            <h1>Microgrid Alert</h1>
            <h2>Summary:</h2>
            {generate_alert_summary(alert_records)}
            <h2>Alert Details:</h2>
            {generate_alert_table(alert_records, displayed_rows)}
            <hr>
            <div class="footer" style="text-align: center; font-size: 12px; color: #777;">
                <p>&copy; 2024 Frequenz Energy-as-a-Service GmbH. All rights reserved.</p>
                <p><a href="{html.escape(notebook_url)}">Manage Alert Preferences</a></p>
            </div>
        </body>
    </html>
    """
