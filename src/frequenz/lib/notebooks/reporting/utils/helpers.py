# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH


"""Energy flow derivation and composition helpers.

This module provides utilities to calculate and append derived energy flow
metrics—such as production excess, grid feed-in, battery charging, and
self-consumption—from given production and consumption columns.

Typical usage involves:
  1. Summing production and consumption sources.
  2. Calculating derived flows (excess, battery, grid, self-use).
  3. Returning an enriched DataFrame for reporting or visualization.

Main Functions:
  - `_get_numeric_series()`: Safe numeric extraction or zero fallback.
  - `_sum_cols()`: Elementwise sum of multiple numeric columns.
  - `add_energy_flows()`: Derive and append core energy flow metrics.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import pandas as pd
import plotly.express as px

from frequenz.lib.notebooks.reporting.metrics.reporting_metrics import (
    asset_production,
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


# pylint: disable=too-many-arguments, too-many-locals
def add_energy_flows(
    df: pd.DataFrame,
    production_cols: list[str] | None = None,
    consumption_cols: list[str] | None = None,
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
        battery_cols: optional column names representing signed battery power.
            Positive values indicate charging, negative values indicate discharging.
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
    resolved_battery_cols = [
        col for col in (battery_cols or []) if _column_has_data(df_flows, col)
    ]

    battery_power_series = _sum_cols(df_flows, resolved_battery_cols)
    battery_charge_series = (
        battery_power_series.reindex(df_flows.index).fillna(0.0).clip(lower=0.0)
    )

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

    df_flows = df_flows.drop(
        columns=["production_total", "consumption_total"], errors="ignore"
    )
    return df_flows


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
    # --- Default palette ---
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

    # --- Assign user-defined colors first ---
    if color_dict:
        for c, v in color_dict.items():
            if c in cols:
                rgba = to_rgba_str(v)
                final[c] = rgba
                used.add(rgba)

    # --- Assign remaining colors from palette ---
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
