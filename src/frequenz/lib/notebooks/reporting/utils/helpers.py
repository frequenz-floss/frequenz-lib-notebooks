# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH
# pylint: disable=too-many-lines

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

import os
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal, Mapping, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import matplotlib.colors as mcolors
import pandas as pd
import plotly.express as px
import yaml
from frequenz.client.assets import AssetsApiClient
from frequenz.client.common.microgrid import MicrogridId

from frequenz.data.microgrid import MicrogridConfig
from frequenz.lib.notebooks._credentials import resolve_credentials
from frequenz.lib.notebooks.reporting.metrics.reporting_metrics import (
    DEFAULT_RESOLUTION,
    GridDataSource,
    asset_production,
    battery_power_flows,
    consumption,
    grid_consumption,
    grid_directions,
    grid_feed_in,
    production_excess,
    production_self_consumption,
    production_self_share,
    production_self_usage,
)
from frequenz.lib.notebooks.reporting.utils.colors import COLOR_DICT
from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper

SummaryPeriod = Literal["daily", "weekly", "monthly", "yearly"]


@dataclass(frozen=True)
class SummaryAggregateData:
    """Prepared data for summary aggregate plotting."""

    summary: pd.DataFrame
    positive: pd.DataFrame
    negative: pd.DataFrame


_SUMMARY_PERIOD_RULES: dict[SummaryPeriod, str] = {
    "daily": "1D",
    "weekly": "1W",
    "monthly": "1MS",
    "yearly": "1YS",
}

AggregatedComponentConfig = Mapping[str, tuple[str, str]]

DEFAULT_AGGREGATED_COMPONENT_CONFIG: AggregatedComponentConfig = {
    "battery": ("battery_power_flow", "Battery #"),
    "pv": ("pv_asset_production", "PV #"),
    "chp": ("chp_asset_production", "CHP #"),
    "wind": ("wind_asset_production", "Wind #"),
}


def _extract_component_id(component_id: Any) -> str | None:
    """Extract the numeric part from a component identifier."""
    digits = "".join(ch for ch in str(component_id) if ch.isdigit())
    return digits or None


def _resolved_assets_api_config(
    server_url: str | None,
    auth_key: str | None,
    sign_secret: str | None,
) -> tuple[str, str | None, str | None]:
    """Resolve Assets API connection settings from args and environment."""
    resolved_server_url = server_url or os.getenv("ASSETS_API_URL")
    if auth_key is not None or sign_secret is not None:
        resolved_auth_key = auth_key
        resolved_sign_secret = sign_secret
    else:
        resolved_auth_key, resolved_sign_secret = resolve_credentials(
            os.environ, "ASSETS_API"
        )

    if not resolved_server_url:
        raise ValueError(
            "Assets API URL not configured. Set ASSETS_API_URL or pass server_url."
        )

    return resolved_server_url, resolved_auth_key, resolved_sign_secret


def _component_display_name(component: Any) -> tuple[str, str] | None:
    """Extract a component-id/display-name pair from an asset component."""
    component_key = _extract_component_id(getattr(component, "id", None))
    component_name = getattr(component, "name", None)
    if component_key and component_name:
        return component_key, str(component_name)
    return None


async def get_meter_display_names(
    microgrid_id: int,
    *,
    server_url: str | None = None,
    auth_key: str | None = None,
    sign_secret: str | None = None,
) -> dict[str, str]:
    """Fetch component display names asynchronously from the Assets API."""
    (
        resolved_server_url,
        resolved_auth_key,
        resolved_sign_secret,
    ) = _resolved_assets_api_config(server_url, auth_key, sign_secret)

    client = AssetsApiClient(
        server_url=resolved_server_url,
        auth_key=resolved_auth_key,
        sign_secret=resolved_sign_secret,
    )
    components = await client.list_microgrid_electrical_components(
        MicrogridId(microgrid_id)
    )

    meter_names: dict[str, str] = {}
    for component in components:
        component_pair = _component_display_name(component)
        if component_pair is not None:
            component_id, component_name = component_pair
            meter_names[component_id] = component_name

    return meter_names


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
        pd.Series: The timestamp Series converted to the requested timezone.

    Raises:
        ValueError: If ``ts`` is not a pandas Series.
    """
    if not isinstance(ts, pd.Series):
        raise ValueError("Input must be a pandas Series")

    # Localize naive timestamps
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(assume_tz)

    result = ts.dt.tz_convert(target_tz)
    return cast(pd.Series, result)


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def label_component_columns(
    df: pd.DataFrame,
    mcfg: MicrogridConfig,
    column_battery: str = "battery",
    column_pv: str = "pv",
    column_chp: str = "chp",
    column_ev: str = "ev",
    column_wind: str = "wind",
    component_display_names: Mapping[str, str] | None = None,
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
        component_display_names: Optional mapping from numeric component IDs to
            human-readable display names fetched from the Assets API.

    Returns:
        Tuple containing the renamed DataFrame and the list of applied labels
    """
    # Numeric component columns present in df
    single_components = [str(c) for c in df.columns if str(c).isdigit()]
    available_types = set(mcfg.component_types())

    # From config (empty set if missing). Union of all ID categories, since a
    # component type can have meter, inverter, and component IDs configured
    # at once, and each should be labeled if present in the DataFrame.
    def ids_if_available(t: str) -> set[str]:
        if t not in available_types:
            return set()
        ids: set[str] = set()
        for category in ("meter", "inverter", "component"):
            category_ids = mcfg.component_type_ids(t, component_category=category)
            ids.update(str(x) for x in category_ids or [])
        return ids

    battery_ids = ids_if_available(column_battery)
    pv_ids = ids_if_available(column_pv)
    chp_ids = ids_if_available(column_chp)
    ev_ids = ids_if_available(column_ev)
    wind_ids = ids_if_available(column_wind)

    display_names = component_display_names or {}

    def display_label(prefix: str, component_id: str) -> str:
        component_name = display_names.get(component_id)
        suffix = f" - {component_name}" if component_name else ""
        return f"{prefix} #{component_id}{suffix}"

    rename: dict[str, str] = {}
    rename.update(
        {
            c: display_label(column_battery.capitalize(), c)
            for c in single_components
            if c in battery_ids
        }
    )
    rename.update(
        {
            c: display_label(column_pv.upper(), c)
            for c in single_components
            if c in pv_ids
        }
    )
    rename.update(
        {
            c: display_label(column_ev.upper(), c)
            for c in single_components
            if c in ev_ids
        }
    )
    rename.update(
        {
            c: display_label(column_chp.upper(), c)
            for c in single_components
            if c in chp_ids
        }
    )
    rename.update(
        {
            c: display_label(column_wind.capitalize(), c)
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
        component_types: List of component types (e.g. ["pv", "wind", "chp", "battery"])
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
        "wind": ["wind_asset_production"],
    }

    production_components = set(component_column_map.keys()) - {"battery"}

    # Columns that should be included if ANY production component exists
    universal_production_cols = [
        "production_self_use",
        "grid_feed_in",
        "production_self_share",
        "production_self_usage",
    ]

    # Columns available ONLY when battery exists
    battery_dependent_cols = [
        "production_to_battery",
        "grid_to_battery",
        "battery_to_grid",
        "battery_to_consumption",
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
    grid_energy_consumed_cols: list[str] | None = None,
    grid_energy_delivered_cols: list[str] | None = None,
    battery_cols: list[str] | None = None,
    grid_data_source: GridDataSource = "energy",
    resolution: timedelta = DEFAULT_RESOLUTION,
) -> pd.DataFrame:
    """Compute and add derived energy flow metrics to the DataFrame.

    This function aggregates production and consumption data, derives energy flow
    relationships such as grid feed-in, battery charging, and self-consumption,
    and appends these computed columns to the given DataFrame. Directional grid
    columns are used whenever present, including when all their values are zero.
    Grid energy deltas are converted to average power before any flow metrics
    are calculated, so they remain compatible with the other power inputs.

    Args:
        df: Input DataFrame containing production, consumption, and optionally
            battery power data.
        production_cols: list of column names representing production sources.
        consumption_cols: list of column names representing consumption sources.
        grid_energy_consumed_cols: AC active energy consumed from the grid.
        grid_energy_delivered_cols: AC active energy delivered to the grid.
        battery_cols: optional list of column names for battery charging power. If None,
            battery-related flows are set to zero.
        grid_data_source: Select directional energy deltas (the default) or
            signed grid power.
        resolution: Reporting period duration, required for "energy" mode.

    Returns:
        A DataFrame including additional columns:
            - "production_excess": Production exceeding consumption.
            - "grid_feed_in": Portion of excess fed into the grid.
            - "grid": Grid import with feed-in values removed when a measured
              grid column is provided.
            - "production_self_use": Self-consumed portion of production.
            - "production_to_battery": Production energy sent to the battery.
            - "grid_to_battery": Grid energy sent to the battery.
            - "battery_to_grid": Battery discharge exported to the grid.
            - "battery_to_consumption": Battery discharge used by consumption.
            - "production_self_share": Share of production that is self-consumed
              (self-consumed / total production).
            - "production_self_usage": Share of consumption covered by
              self-production (self-consumed / total consumption).

    Raises:
        ValueError: If grid_data_source is unknown, if energy mode is missing
            either directional energy column, or if its resolution is invalid.
    """
    df_flows = df.copy()

    # Normalize production, consumption and battery columns by removing None entries
    resolved_production_cols = [
        col for col in (production_cols or []) if _column_has_data(df_flows, col)
    ]
    resolved_consumption_cols = [
        col for col in (consumption_cols or []) if _column_has_data(df_flows, col)
    ]
    resolved_grid_energy_consumed_cols = [
        col for col in (grid_energy_consumed_cols or []) if col in df_flows.columns
    ]
    resolved_grid_energy_delivered_cols = [
        col for col in (grid_energy_delivered_cols or []) if col in df_flows.columns
    ]
    resolved_battery_cols = [
        col for col in (battery_cols or []) if _column_has_data(df_flows, col)
    ]

    battery_power_series = _sum_cols(df_flows, resolved_battery_cols)
    if grid_data_source not in ("power", "energy"):
        raise ValueError(f"Unknown grid data source: {grid_data_source}")
    if grid_data_source == "energy" and not (
        resolved_grid_energy_consumed_cols and resolved_grid_energy_delivered_cols
    ):
        raise ValueError(
            "Energy grid data requires both consumed and delivered columns."
        )

    has_grid_measurement = "grid" in df_flows.columns
    if grid_data_source == "energy":
        has_grid_measurement = bool(
            resolved_grid_energy_consumed_cols and resolved_grid_energy_delivered_cols
        )

    if has_grid_measurement:
        grid_consumption_series, grid_feed_in_series = grid_directions(
            data_source=grid_data_source,
            grid=_get_numeric_series(df_flows, "grid"),
            grid_import=_sum_cols(df_flows, resolved_grid_energy_consumed_cols),
            grid_export=_sum_cols(df_flows, resolved_grid_energy_delivered_cols),
            resolution=resolution,
        )
    else:
        grid_consumption_series = pd.Series(0.0, index=df_flows.index)
        grid_feed_in_series = pd.Series(0.0, index=df_flows.index)

    # Compute total asset production
    asset_production_cols: list[str] = []
    for col in resolved_production_cols:
        series = _get_numeric_series(
            df_flows,
            col,
        )
        asset_series = asset_production(
            series,
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

    if resolved_consumption_cols:
        df_flows["consumption_total"] = _sum_cols(df_flows, consumption_series_cols)
    else:
        # When no explicit consumption columns exist, infer it from grid info.
        df_flows["consumption_total"] = consumption(
            grid_consumption_series,
            grid_feed_in_series,
            production=_sum_cols(
                df_flows, resolved_production_cols
            ),  # raw production to have the negative production values
            battery=battery_power_series,
        )

        # Set consumption to total consumption if no explicit consumption columns exist
        df_flows["consumption"] = df_flows["consumption_total"].copy()

    # Surplus vs. consumption (production is already positive because of the above cleaning)
    df_flows["production_excess"] = production_excess(
        df_flows["production_total"] * -1,
        df_flows["consumption_total"],
    )

    # Split excess into battery vs. grid
    df_flows["grid_feed_in"] = grid_feed_in(
        grid_feed_in_series if has_grid_measurement else None,
        # To convert positive production back to PSC format (where production is negative)
        df_flows["production_total"] * -1,
        df_flows["consumption_total"],
        battery=battery_power_series,
    )

    # The power-mode grid series carries both directions: imports are positive
    # and feed-in is negative. Retain only imports in the raw grid column now
    # that feed-in has been split into its own column.
    if grid_data_source == "power" and "grid" in df_flows.columns:
        df_flows["grid"] = grid_consumption_series

    # If no production columns exist, set self-consumption metrics to zero
    if asset_production_cols:
        # Use total production for self-consumption instead of asset_production
        # (which may not exist)
        df_flows["production_self_use"] = production_self_consumption(
            df_flows["production_total"] * -1,
            df_flows["consumption_total"],
        )
        df_flows["production_self_share"] = production_self_share(
            df_flows["production_total"] * -1,
            df_flows["consumption_total"],
        )
        df_flows["production_self_usage"] = production_self_usage(
            df_flows["production_total"] * -1,
            df_flows["consumption_total"],
        )
    else:
        df_flows["production_self_use"] = 0.0
        df_flows["production_self_share"] = 0.0
        df_flows["production_self_usage"] = 0.0

    df_flows["grid_consumption"] = grid_consumption(
        grid_consumption_series if has_grid_measurement else None,
        # To convert positive production back to PSC format (where production is negative)
        df_flows["production_total"] * -1,
        df_flows["consumption_total"],
        battery_power_series,
    )

    (
        df_flows["production_to_battery"],
        df_flows["grid_to_battery"],
        df_flows["battery_to_grid"],
        df_flows["battery_to_consumption"],
    ) = battery_power_flows(
        grid_import=df_flows["grid_consumption"],
        grid_export=df_flows["grid_feed_in"],
        production=df_flows["production_total"],
        consumption=df_flows["consumption_total"],
        battery=battery_power_series,
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
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, KeyError):
        warnings.warn(
            f"Unknown timezone '{timezone_name}', falling back to UTC.",
            RuntimeWarning,
        )
        tz = ZoneInfo("UTC")

    return datetime.combine(input_date, time.min).replace(tzinfo=tz)


def normalize_date_for_reporting(
    input_date: date | datetime, timezone_name: str = "UTC"
) -> datetime:
    """Return midnight for past dates or current time for today.

    If the input date is today (in the target timezone), this returns the
    current time in that timezone. If the date is in the future, a ValueError
    is raised to prevent selecting future dates. For past dates, midnight is
    returned via ``set_date_to_midnight``.

    Args:
        input_date: Date or datetime object to evaluate.
        timezone_name: Name of the target timezone (e.g., "Europe/Berlin").
            Defaults to "UTC". Falls back to UTC if the timezone name
            is invalid.

    Returns:
        A timezone-aware datetime object representing the current time
        when the input date is today, or midnight for past dates.

    Raises:
        ValueError: If the input date is in the future.
    """
    input_day = input_date.date() if isinstance(input_date, datetime) else input_date

    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, KeyError):
        warnings.warn(
            f"Unknown timezone '{timezone_name}', falling back to UTC.",
            RuntimeWarning,
        )
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)
    today = now.date()

    if input_day > today:
        raise ValueError("Future dates can't be selected.")
    if input_day == today:
        return now

    return set_date_to_midnight(input_date, timezone_name)


AggFuncLiteral = Literal[
    "mean",
    "sum",
    "count",
    "min",
    "max",
    "median",
    "std",
    "var",
]


def long_to_wide(
    df: pd.DataFrame,
    *,
    time_col: str | pd.Index = "Timestamp",
    category_col: str | None = "Battery",
    value_col: str | None = "Battery Throughput",
    sum_col_name: str | None = None,
    aggfunc: AggFuncLiteral = "sum",
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
    values. Default colors are sourced from the in-code color dictionary
    and can be overridden via `color_dict`. Remaining columns are assigned
    distinct colors from a chosen palette, ensuring no duplicates.

    Args:
        cols: List of column names or category labels to assign colors to.
        color_dict: Optional dictionary of color mappings to override defaults.
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

    # First assign default colors, then override with user-provided
    merged_color_dict = COLOR_DICT
    if color_dict:
        merged_color_dict = {**merged_color_dict, **color_dict}

    for c, v in merged_color_dict.items():
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


def _date_index(index: pd.Index) -> pd.Index:
    """Return an index containing the date part of a datetime-like index."""
    return pd.Index(pd.DatetimeIndex(index).date)


def prepare_summary_data(
    df: pd.DataFrame, period: SummaryPeriod = "monthly"
) -> SummaryAggregateData:
    """Aggregate power data into MWh totals for summary plotting.

    Args:
        df: Input time-series DataFrame with a ``DatetimeIndex`` or a
            ``timestamp`` column.
        period: Aggregation period: ``daily``, ``weekly``, ``monthly``, or
            ``yearly``.

    Returns:
        The aggregated summary data split into positive and negative values.

    Raises:
        TypeError: If ``df`` does not have a timestamp index or column.
        ValueError: If fewer than two rows are provided or ``period`` is unknown.
    """
    source_input = df.copy()
    if not isinstance(source_input.index, pd.DatetimeIndex):
        if "timestamp" not in source_input.columns:
            raise TypeError("DataFrame must have a DatetimeIndex or timestamp column.")
        source_input = source_input.set_index(pd.to_datetime(source_input["timestamp"]))
    if len(source_input.index) < 2:
        raise ValueError(
            "At least two data points are required for summary aggregation."
        )
    if period not in _SUMMARY_PERIOD_RULES:
        raise ValueError(
            f"Unsupported summary period: {period!r}. "
            f"Expected one of {sorted(_SUMMARY_PERIOD_RULES)}."
        )

    mapper = ColumnMapper.from_default(locale="de")
    labels = mapper.canonical_to_display
    source_df = mapper.to_canonical(source_input).select_dtypes(include="number")
    if "pv" in source_df.columns and "pv_asset_production" not in source_df.columns:
        source_df["pv_asset_production"] = source_df["pv"]
    resolution = (source_input.index[1] - source_input.index[0]).total_seconds()
    kw_to_mwh = resolution / 3600 / 1000
    resample_rule = _SUMMARY_PERIOD_RULES[period]
    positive_columns = [
        "grid_consumption",
        "mid_consumption",
        "battery_charge",
    ]
    negative_columns = [
        "grid_feed_in",
        "battery_discharge",
        "pv_asset_production",
        "chp_asset_production",
        "wind_asset_production",
    ]

    def _period_sum(series: pd.Series) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors="coerce")
        result: pd.Series = numeric_series.resample(resample_rule).sum()
        result = result.mul(kw_to_mwh)
        result.index = _date_index(result.index)
        return result

    index = _date_index(source_df.resample(resample_rule).sum(numeric_only=True).index)
    positive = pd.DataFrame(index=index)
    negative = pd.DataFrame(index=index)

    for column in positive_columns:
        if column in source_df.columns:
            positive[labels[column]] = _period_sum(source_df[column])

    for column in negative_columns:
        if column in source_df.columns:
            negative[labels[column]] = _period_sum(-source_df[column].abs())

    positive = positive.loc[:, positive.abs().sum(axis=0) > 0]
    negative = negative.loc[:, negative.abs().sum(axis=0) > 0]
    summary = pd.concat([positive, negative], axis=1)

    return SummaryAggregateData(summary=summary, positive=positive, negative=negative)


def fill_aggregated_component_columns(
    df: pd.DataFrame,
    component_types: list[str],
    config: AggregatedComponentConfig | None = None,
) -> pd.DataFrame:
    """Populate missing aggregate columns by summing labeled component columns.

    Args:
        df: Input DataFrame potentially containing aggregated component columns
            (e.g., "battery_power_flow") and labeled individual component columns
            (e.g., "Battery #1", "Battery #2").
        component_types: List of component types to consider for aggregation
            (e.g., ["battery", "pv", "chp", "wind"]).

        config: Mapping of component types to tuples containing the aggregated
            column name and the prefix used to identify individual component
            columns.

    Returns:
        DataFrame with missing aggregated component columns filled in by summing
        the corresponding individual component columns.
    """
    config = config or DEFAULT_AGGREGATED_COMPONENT_CONFIG
    normalized_types = {comp.lower() for comp in component_types}

    for comp_type, (agg_col, prefix) in config.items():
        if comp_type not in normalized_types or agg_col not in df.columns:
            continue

        component_cols = [col for col in df.columns if col.startswith(prefix)]

        # Only proceed if there are components to sum and missing values to fill
        if component_cols and df[agg_col].isna().any():
            # min_count=1 ensures all-NaN rows stay NaN instead of becoming 0
            summed = df[component_cols].sum(axis=1, skipna=True, min_count=1)

            # Fill only the holes
            df[agg_col] = df[agg_col].fillna(summed)

    return df
