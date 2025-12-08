# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Energy flow and configuration utilities for microgrid analysis and reporting.

This module provides helper functions for:
  - Safe numeric extraction and aggregation from pandas DataFrames.
  - Loading and validating YAML-based configuration files.
  - Labeling microgrid component columns using configuration metadata.
  - Computing derived energy flow metrics such as:
      * Production excess
      * Battery charge utilization
      * Grid feed-in
      * Self-consumption and self-share
  - Formatting and timezone conversion utilities for reporting.

These utilities are primarily used in energy analytics pipelines and
microgrid reporting notebooks to ensure consistent data preprocessing,
metric calculation, and standardized output structures.

Functions:
    _get_numeric_series: Safely extract a numeric Series or return zeros if missing.
    _sum_cols: Safely sum multiple numeric columns.
    load_config: Load and validate a YAML configuration file.
    _fmt_to_de_system: Format numbers using German-style decimal conventions.
    _convert_timezone: Convert a DataFrame timestamp column to a target timezone.
    label_component_columns: Rename numeric component columns using MicrogridConfig.
    get_energy_report_columns: Determine relevant columns for energy reporting.
    add_energy_flows: Compute derived production, battery, and grid metrics.

Notes:
    - These helpers are designed for internal use and assume well-structured
      DataFrames with datetime indices or timestamp columns.
    - All numeric outputs are returned as float64 Series to ensure consistency.
"""


from __future__ import annotations

import warnings
from datetime import date, datetime, time
from typing import Any

import matplotlib.colors as mcolors
import pandas as pd
import plotly.express as px
import pytz
import yaml

from frequenz.data.microgrid.config import MicrogridConfig
from frequenz.lib.notebooks.reporting.metrics.reporting_metrics import (
    asset_production,
    grid_consumption,
    grid_feed_in,
    production_excess,
    production_excess_in_bat,
    production_self_consumption,
    production_self_share,
)


def _get_numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    """Safely extract a numeric Series or return zeros if missing.

    Ensures consistent numeric handling even when the requested column
    does not exist or is None. Returns a zero-filled Series aligned to
    the DataFrame index when the column is unavailable.

    Args:
        df: Input DataFrame from which to extract the column.
        col: Column name to retrieve. If None or missing, zeros are returned.

    Returns:
        A float64 Series aligned to the input index.
    """
    if col is None:
        series = pd.Series(0.0, index=df.index, dtype="float64")
    else:
        raw = df.reindex(columns=[col], fill_value=0)[col]
        series = pd.to_numeric(raw, errors="coerce").fillna(0.0).astype("float64")
    return series


def _sum_cols(df: pd.DataFrame, cols: list[str] | None) -> pd.Series:
    """Safely sum multiple numeric columns into a single Series.

    Ensures robust aggregation even when some columns are missing or None.
    Missing columns are treated as zero-valued Series aligned to the DataFrame index.

    Args:
        df: Input DataFrame containing the columns to be summed.
        cols: list of column names to sum. If empty, returns a zero-filled Series.

    Returns:
        A float64 Series representing the elementwise sum of all specified columns.
        Missing or invalid columns are treated as zeros.
    """
    if not cols:
        return pd.Series(0.0, index=df.index, dtype="float64")

    # Safely extract each column as a numeric Series then sum row-wise
    series_list = [_get_numeric_series(df, c) for c in cols]
    return pd.concat(series_list, axis=1).sum(axis=1).astype("float64")


def _column_has_data(df: pd.DataFrame, col: str | None) -> bool:
    """Return True when the column exists and has at least one non-zero value."""
    if col is None or col not in df.columns:
        return False

    series = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
    if series.empty or not series.notna().any():
        return False

    return not series.fillna(0).eq(0).all()


def load_config(path: str) -> dict[str, Any]:
    """
    Load a YAML config file and return it as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Configuration values as a dictionary.

    Raises:
        TypeError: If the YAML root element is not a mapping (dict).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"YAML root must be a mapping (dict), got {type(data).__name__}"
        )

    return data


def fmt_to_de_system(x: float) -> str:
    """Format a number using German-style decimal and thousands separators.

    The function formats the number with two decimal places, using a comma
    as the decimal separator and a dot as the thousands separator.

    Args:
        x: The number to format.

    Returns:
        The formatted string with German number formatting applied.

    Example:
        >>> _fmt_to_de_system(12345.6789)
        '12.345,68'
    """
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def convert_timezone(
    ts: pd.Series,
    target_tz: str = "Europe/Berlin",
    assume_tz: str = "UTC",
) -> pd.Series:
    """Convert a datetime Series to a target timezone.

    If the Series contains timezone-naive datetimes, they are first localized to
    ``assume_tz`` before converting to ``target_tz``.

    Args:
        ts: Input Series containing the datetime values.
        target_tz: Timezone name to convert the Series to.
            Defaults to ``"Europe/Berlin"``.
        assume_tz: Timezone to assume for naive datetimes.
            Defaults to ``"UTC"``.

    Returns:
        pd.DataFrame: A copy of the DataFrame with the converted datetime column.

    Raises:
        ValueError: If ``column_timestamp`` is not present in ``df``.
    """
    if not isinstance(ts, pd.Series):
        raise ValueError("Input must be a pandas Series")

    # Localize naive timestamps
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(assume_tz)

    return ts.dt.tz_convert(target_tz)


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def label_component_columns(
    df: pd.DataFrame,
    mcfg: MicrogridConfig,
    column_battery: str = "battery",
    column_pv: str = "pv",
    column_chp: str = "chp",
    column_ev: str = "ev",
    column_wind: str = "wind",
) -> tuple[pd.DataFrame, list[str]]:
    """Rename numeric single-component columns to labeled names.

    Numeric string column names like ``"14"`` are converted to
    ``"Battery #14"``, ``"PV #14"``, ``"CHP #14"`` or ``"EV #14"`` based on
    the component IDs provided by ``mcfg.component_type_ids(...)``

    Args:
        df: Input DataFrame with numeric string column names.
        mcfg: Configuration with ``_component_types_cfg`` mapping component types to a
            ``meter`` iterable of numeric IDs.
        column_battery: Key name for battery component type.
        column_pv: Key name for PV component type.
        column_chp: Key name for CHP component type.
        column_ev: Key name for EV component type.
        column_wind: Key name for wind component type.

    Returns:
        Tuple containing the renamed DataFrame and the list of applied labels
    """
    # Numeric component columns present in df
    single_components = [str(c) for c in df.columns if str(c).isdigit()]
    available_types = set(mcfg.component_types())

    # From config (empty set if missing)
    def ids_if_available(t: str) -> set[str]:
        return (
            {str(x) for x in mcfg.component_type_ids(t)}
            if t in available_types
            else set()
        )

    battery_ids = ids_if_available(column_battery)
    pv_ids = ids_if_available(column_pv)
    chp_ids = ids_if_available(column_chp)
    ev_ids = ids_if_available(column_ev)
    wind_ids = ids_if_available(column_wind)

    rename: dict[str, str] = {}
    rename.update(
        {
            c: f"{column_battery.capitalize()} #{c}"
            for c in single_components
            if c in battery_ids
        }
    )
    rename.update(
        {c: f"{column_pv.upper()} #{c}" for c in single_components if c in pv_ids}
    )
    rename.update(
        {c: f"{column_ev.upper()} #{c}" for c in single_components if c in ev_ids}
    )
    rename.update(
        {c: f"{column_chp.upper()} #{c}" for c in single_components if c in chp_ids}
    )
    rename.update(
        {
            c: f"{column_wind.capitalize()} #{c}"
            for c in single_components
            if c in wind_ids
        }
    )

    return df.rename(columns=rename), list(rename.values())


def get_energy_report_columns(
    component_types: list[str], single_components: list[str]
) -> list[str]:
    """Build the list of dataframe columns for the energy report.

    The selected columns depend on the available component types.

    Args:
        component_types: List of component types (e.g. ["pv", "battery"])
        single_components: Extra component columns to always include.

    Returns:
        The full list of dataframe columns.
    """
    # Base columns
    energy_report_df_cols = [
        "timestamp",
        "grid_consumption",
        "mid_consumption",
    ] + single_components

    # Map component types to the columns they enable
    component_column_map = {
        "battery": ["battery_power_flow"],
        "pv": ["pv_asset_production"],
        "chp": ["chp_asset_production"],
        "ev": ["ev_asset_production"],
        "wind": ["wind_asset_production"],
    }

    production_components = set(component_column_map.keys()) - {"battery"}

    # Columns that should be included if ANY production component exists
    universal_production_cols = [
        "production_self_use",
        "grid_feed_in",
        "production_self_share",
    ]

    # Columns available ONLY when battery exists
    battery_dependent_cols = [
        "production_excess_in_bat",
    ]

    # Check if any production component is present
    has_production = any(comp in production_components for comp in component_types)
    has_battery = "battery" in component_types

    # Add the universal production columns ONLY if production exists
    if has_production:
        energy_report_df_cols.extend(universal_production_cols)

    # Add battery-dependent production columns
    if has_battery:
        energy_report_df_cols.extend(battery_dependent_cols)

    # Add component-specific columns
    for comp in component_types:
        if comp in component_column_map:
            energy_report_df_cols.extend(component_column_map[comp])

    return energy_report_df_cols


# pylint: disable=too-many-arguments, too-many-locals
def add_energy_flows(
    df: pd.DataFrame,
    production_cols: list[str] | None = None,
    consumption_cols: list[str] | None = None,
    grid_cols: list[str] | None = None,
    battery_cols: list[str] | None = None,
    production_is_positive: bool = False,
) -> pd.DataFrame:
    """Compute and add derived energy flow metrics to the DataFrame.

    This function aggregates production and consumption data, derives energy flow
    relationships such as grid feed-in, battery charging, and self-consumption,
    and appends these computed columns to the given DataFrame. Columns that are
    specified but missing or contain only null/zero values are ignored.

    Args:
        df: Input DataFrame containing production, consumption, and optionally
            battery power data.
        production_cols: list of column names representing production sources.
        consumption_cols: list of column names representing consumption sources.
        grid_cols: list of column names representing grid import/export.
        battery_cols: optional list of column names for battery charging power. If None,
            battery-related flows are set to zero.
        production_is_positive: Whether production values are already positive.
            If False, `production` is inverted before clipping.

    Returns:
        A DataFrame including additional columns:
            - "production_excess": Production exceeding consumption.
            - "production_excess_in_bat": Portion of excess stored in the battery.
            - "grid_feed_in": Portion of excess fed into the grid.
            - "production_self_use": Self-consumed portion of production.
            - "production_self_share": Share of consumption covered by self-production.
    """
    df_flows = df.copy()

    # Normalize production, consumption and battery columns by removing None entries
    resolved_production_cols = [
        col for col in (production_cols or []) if _column_has_data(df_flows, col)
    ]
    resolved_consumption_cols = [
        col for col in (consumption_cols or []) if _column_has_data(df_flows, col)
    ]
    resolved_grid_cols = [
        col for col in (grid_cols or []) if _column_has_data(df_flows, col)
    ]
    resolved_battery_cols = [
        col for col in (battery_cols or []) if _column_has_data(df_flows, col)
    ]

    battery_power_series = _sum_cols(df_flows, resolved_battery_cols)
    battery_charge_series = (
        battery_power_series.reindex(df_flows.index).fillna(0.0).clip(lower=0.0)
    )
    grid_power_series = _sum_cols(df_flows, resolved_grid_cols)

    # Compute total asset production
    asset_production_cols: list[str] = []
    for col in resolved_production_cols:
        series = _get_numeric_series(
            df_flows,
            col,
        )
        asset_series = asset_production(
            series,
            production_is_positive=production_is_positive,
        )
        asset_col_name = f"{col}_asset_production"
        df_flows[asset_col_name] = asset_series
        asset_production_cols.append(asset_col_name)

    df_flows["production_total"] = _sum_cols(df_flows, asset_production_cols)

    # Compute total consumption
    consumption_series_cols: list[str] = []
    for col in resolved_consumption_cols:
        df_flows[col] = _get_numeric_series(df_flows, col)
        consumption_series_cols.append(col)

    df_flows["consumption_total"] = _sum_cols(df_flows, consumption_series_cols)

    # Surplus vs. consumption (production is already positive because of the above cleaning)
    df_flows["production_excess"] = production_excess(
        df_flows["production_total"],
        df_flows["consumption_total"],
        production_is_positive=True,
    )

    # Battery charging power (optional)
    df_flows["production_excess_in_bat"] = production_excess_in_bat(
        df_flows["production_total"],
        df_flows["consumption_total"],
        battery=battery_charge_series,
        production_is_positive=True,
    )

    # Split excess into battery vs. grid
    df_flows["grid_feed_in"] = grid_feed_in(
        df_flows["production_total"],
        df_flows["consumption_total"],
        battery=battery_charge_series,
        production_is_positive=True,
    )

    # If no production columns exist, set self-consumption metrics to zero
    if asset_production_cols:
        # Use total production for self-consumption instead of asset_production
        # (which may not exist)
        df_flows["production_self_use"] = production_self_consumption(
            df_flows["production_total"],
            df_flows["consumption_total"],
            production_is_positive=True,
        )
        df_flows["production_self_share"] = production_self_share(
            df_flows["production_total"],
            df_flows["consumption_total"],
            production_is_positive=True,
        )
    else:
        df_flows["production_self_use"] = 0.0
        df_flows["production_self_share"] = 0.0

    # Add grid consumption column - grid is later renamed as grid_consumption
    if "grid" not in df_flows.columns:
        df_flows["grid"] = grid_consumption(
            grid_power_series,
            # To convert positive production back to PSC format (where production is negative)
            df_flows["production_total"] * -1,
            df_flows["consumption_total"],
            battery_power_series,
        )

    df_flows = df_flows.drop(
        columns=["production_total", "consumption_total"], errors="ignore"
    )
    return df_flows


def set_date_to_midnight(
    input_date: date | datetime, timezone_name: str = "UTC"
) -> datetime:
    """Return a timezone-aware datetime set to midnight of the given date.

    Converts a date or datetime into a midnight timestamp localized to
    the specified timezone. If the input is already a datetime, only the
    date portion is used.

    Args:
        input_date: Date or datetime object to normalize to midnight.
        timezone_name: Name of the target timezone (e.g., "Europe/Berlin").
            Defaults to "UTC". Falls back to UTC if the timezone name
            is invalid.

    Returns:
        A timezone-aware datetime object representing midnight of the
        given date in the specified timezone.
    """
    if isinstance(input_date, datetime):
        input_date = input_date.date()

    try:
        tz = pytz.timezone(timezone_name)
    except pytz.UnknownTimeZoneError:
        warnings.warn(
            f"Unknown timezone '{timezone_name}', falling back to UTC.",
            RuntimeWarning,
        )
        tz = pytz.UTC

    return tz.localize(datetime.combine(input_date, time.min))


def long_to_wide(
    df: pd.DataFrame,
    *,
    time_col: str | pd.Index = "Timestamp",
    category_col: str | None = "Battery",
    value_col: str | None = "Battery Throughput",
    sum_col_name: str | None = None,
    aggfunc: str = "sum",
) -> pd.DataFrame:
    """Convert a long-format DataFrame into wide format with optional aggregation.

    Transforms a long-format dataset (one row per timestamp-category pair)
    into a wide-format table, where each category becomes a separate column.
    Optionally adds a total (sum) column across all categories.

    Args:
        df: Input DataFrame in long format.
        time_col: Column name representing timestamps used as the index in
            the resulting wide table. Defaults to `"Timestamp"`.
        category_col: Column name representing category labels that become
            column headers in the wide table. Defaults to `"Battery"`.
        value_col: Column name representing numeric values to aggregate and
            pivot into columns. Defaults to `"Battery Throughput"`.
        sum_col_name: Optional name for a new column containing the row-wise sum
            of all category columns. If None, defaults to `"<value_col> Sum"`.
        aggfunc: Aggregation function applied when multiple entries exist per
            timestamp-category pair (e.g., `"sum"`, `"mean"`). Defaults to `"sum"`.

    Returns:
        A wide-format DataFrame with one row per timestamp, one column per category,
        and an optional total column representing the aggregated sum across all categories.
    """
    tmp = df.copy()

    wide = tmp.pivot_table(
        index=time_col,  # type: ignore [arg-type]
        columns=category_col,
        values=value_col,
        aggfunc=aggfunc,
    ).sort_index()

    wide.columns.name = None

    if sum_col_name is None:
        sum_col_name = f"{value_col} Sum"
    wide[sum_col_name] = wide.sum(axis=1, numeric_only=True)
    return wide


def build_color_map(
    cols: list[str],
    color_dict: dict[str, str] | None = None,
    palette: list[str] | None = None,
) -> dict[str, str]:
    """Generate a color mapping for columns or categories.

    Creates a mapping from column names (or categorical labels) to color
    values. If user-specified colors are provided via `color_dict`, those
    are applied first. Remaining columns are assigned distinct colors from
    a chosen palette, ensuring no duplicates.

    Args:
        cols: List of column names or category labels to assign colors to.
        color_dict: Optional dictionary of pre-defined color mappings.
            Columns found here are assigned these colors directly.
        palette: Optional list of color codes to use as defaults.
            If None, a combined Plotly qualitative palette is used.

    Returns:
        A dictionary mapping each column or category name to a unique color.
    """
    # Default palette
    if palette is None:
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark2

    def to_rgba_str(color: str) -> str:
        """Convert any color format (hex, rgb, named) to normalized rgba(R,G,B,1) string."""
        try:
            rgba = mcolors.to_rgba(color)  # returns (r,g,b,a) in 0–1 range
            rgba_255 = tuple(int(round(x * 255)) for x in rgba[:3])
            return f"rgba({rgba_255[0]},{rgba_255[1]},{rgba_255[2]},{rgba[3]:.3f})"
        except ValueError:
            # fallback if string isn't recognized (e.g. malformed rgba)
            return color.lower().strip()

    final = {}
    used = set()

    # First assign user-provided colors
    if color_dict:
        for c, v in color_dict.items():
            if c in cols:
                rgba = to_rgba_str(v)
                final[c] = rgba
                used.add(rgba)

    # Then assign defaults, skipping already-used colors
    palette_iter = iter(palette * (len(cols) // len(palette) + 1))
    for c in cols:
        if c in final:
            continue
        for p in palette_iter:
            rgba = to_rgba_str(p)
            if rgba not in used:
                final[c] = rgba
                used.add(rgba)
                break

    return final
