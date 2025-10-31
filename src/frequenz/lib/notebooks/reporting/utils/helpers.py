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

import pandas as pd

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


# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
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
