# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH
"""Helper functions for Microgrid Data Processing Utilities.

This module provides utility functions for preprocessing and analyzing microgrid
data represented in pandas DataFrames. It derives photovoltaic (PV) energy flows
from the given dataframe.

Key Features
------------
- PV Energy Flow Calculations
  Derives PV production, excess, self-consumption, battery charging, and
  grid feed-in metrics, including PV self-consumption share.

Usage
-----
These functions serve as building blocks for energy reporting, data pipelines,
and dashboards that analyze microgrid performance, particularly in hybrid systems
with PV, batteries, and grid interactions.
"""

from typing import Any, List, Optional

import pandas as pd


def _get_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] if present else a zero series aligned to df.index."""
    return (
        df[name]
        if name in df.columns
        else pd.Series(0, index=df.index, dtype="float64")
    )


def _has_prod_signal(prod: pd.Series) -> bool:
    """Return True if the production signal exists and contains any non-zero values."""
    return isinstance(prod, pd.Series) and bool((prod != 0).any())


def _compute_prod(signal: pd.Series, prod_is_positive: bool) -> pd.Series:
    """Normalize any-signed production signal to positive-only production."""
    return (signal if prod_is_positive else -signal).clip(lower=0)


def _compute_excess(prod: pd.Series, consumption: pd.Series) -> pd.Series:
    """Excess production after subtracting consumption."""
    return (prod - consumption).clip(lower=0)


def _compute_in_bat(excess: pd.Series, battery_pos: pd.Series) -> pd.Series:
    """Portion of excess energy routed into the battery (bounded by charge)."""
    return pd.concat([excess, battery_pos], axis=1).min(axis=1)


def _compute_feed_in(excess: pd.Series, in_bat: pd.Series) -> pd.Series:
    """Energy fed into the grid after battery charging."""
    return excess - in_bat


def _compute_self_use(prod: pd.Series, excess: pd.Series) -> pd.Series:
    """Self-consumed production (production minus excess)."""
    return (prod - excess).clip(lower=0)


def _compute_self_share(self_use: pd.Series, consumption: pd.Series) -> pd.Series:
    """Share of consumption covered by self-consumed production (NaN when consumption is 0)."""
    denom = consumption.replace(0, pd.NA)
    return self_use / denom


# pylint: disable=too-many-arguments, too-many-locals
def _add_energy_flows(
    df: pd.DataFrame,
    *,
    prod_col: str = "prod",
    prod_is_positive: bool = True,
    consumption_col: str = "consumption",
    battery_charge_col: str = "battery_pos",
    out_prefix: str = "",
    out_cols: Optional[dict[str, str]] = None,
    assign: bool = True,
) -> pd.DataFrame:
    """Add production-related energy flow columns if production data is present.

    Derives photovoltaic (PV), CHP and other production energy-flow metrics from
    existing columns. If no production signal is present (i.e., the negative production
    column is missing or all zeros), the DataFrame is returned unchanged.

    Args:
      df: Input DataFrame.
      prod_col: Column with raw production signal (either positive or negative).
      prod_is_positive: If True, positive values indicate production.
                        If False, negative values indicate production.
      consumption_col: Column with household consumption (>=0 expected).
      battery_charge_col: Column with battery charge power (>=0 expected).
      out_prefix: Prefix for output column names (e.g., "pv_", "chp_").
      out_cols: Optional mapping of output column names. Keys can include:
        - "prod": Production column name
        - "excess": Excess production column name
        - "in_bat": Battery charging from production column name
        - "feed_in": Grid feed-in column name
        - "self_use": Self-consumed production column name
        - "self_share": Self-consumption share column name
      assign: If True, add new columns to input DataFrame and return it.
              If False, return a new DataFrame with only the new columns.

    Returns:
      DataFrame with added columns (or unchanged if no production signal).

    Raises:
      ValueError: If the input DataFrame is missing any of the required columns
        specified by `prod_col`, `consumption_col`, or `battery_charge_col`.

    Columns added:
      - prod: normalized production as positive values
      - excess: production minus consumption (>=0)
      - in_bat: portion of excess routed into battery (<= battery charge)
      - feed_in: excess after battery charging (to grid)
      - self_use: production used on-site (prod - excess)
      - self_share: self_use / consumption (NaN when consumption is 0)
    """
    # Safe inputs (0 if missing)
    df_flows = df.copy()
    required_columns = [prod_col, consumption_col, battery_charge_col]
    if not set(required_columns).issubset(df_flows.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    raw_prod = _get_series(df_flows, prod_col)
    consumption = _get_series(df_flows, consumption_col)
    battery_pos = _get_series(df_flows, battery_charge_col)

    if not _has_prod_signal(raw_prod):
        return df_flows if assign else pd.DataFrame(index=df.index)

    # compute
    prod = _compute_prod(raw_prod, prod_is_positive)
    excess = _compute_excess(prod, consumption)
    in_bat = _compute_in_bat(excess, battery_pos)
    feed_in = _compute_feed_in(excess, in_bat)
    self_use = _compute_self_use(prod, excess)
    self_share = _compute_self_share(self_use, consumption)

    # name outputs
    default_names = {
        "prod": f"{out_prefix}prod",
        "excess": f"{out_prefix}excess",
        "in_bat": f"{out_prefix}in_bat",
        "feed_in": f"{out_prefix}feed_in",
        "self_use": f"{out_prefix}self_use",
        "self_share": f"{out_prefix}self_share",
    }

    names = {
        k: (out_cols.get(k, v) if out_cols else v) for k, v in default_names.items()
    }
    new_cols = pd.DataFrame(
        {
            names["prod"]: prod,
            names["excess"]: excess,
            names["in_bat"]: in_bat,
            names["feed_in"]: feed_in,
            names["self_use"]: self_use,
            names["self_share"]: self_share,
        },
        index=df.index,
    )

    if assign:
        for c in new_cols.columns:
            df_flows[c] = new_cols[c]
        return df_flows

    return new_cols


# pylint: disable=too-many-arguments, too-many-locals
def add_energy_flows_multi(
    df: pd.DataFrame,
    sources: List[dict[str, Any]],
    *,
    consumption_col: str = "consumption",
    battery_charge_col: str = "battery_pos",
    cascade_consumption: bool = True,
    cascade_battery: bool = True,
    tmp_prefix: str = "__flows_tmp__",
) -> pd.DataFrame:
    """Add energy flow metrics for multiple production sources with cascading logic.

    Applies `_add_energy_flows` sequentially for multiple on-site energy sources
    (e.g., PV, wind, CHP). Each subsequent source uses the **remaining consumption**
    (and optionally remaining battery charging power) left after previous sources
    have been applied. This prevents double-counting and ensures flows reflect
    actual energy allocation priority.

    Args:
        df: Input DataFrame containing production, consumption, and battery columns.
        sources: List of configuration dictionaries for each energy source.
            Each dict must include:
              - `prod_col`: Column name of the production signal.
              - `prod_is_positive`: Whether production values are positive or negative.
            Optional keys:
              - `prefix`: Prefix for output columns (e.g., `"pv_"`, `"wind_"`).
              - `out_cols`: Custom mapping of output column names.
              - `consumption_col`: Override for consumption input column.
              - `battery_charge_col`: Override for battery input column.
        consumption_col: Base column name for consumption. Defaults to `"consumption"`.
        battery_charge_col: Base column name for battery charging. Defaults to `"battery_pos"`.
        cascade_consumption: If True, pass leftover consumption to next source. Defaults to True.
        cascade_battery: If True, pass leftover battery charge to next source. Defaults to True.
        tmp_prefix: Prefix for temporary internal columns, which are removed afterward.

    Returns:
        DataFrame with additional columns for each source, such as:
        `<prefix>prod`, `<prefix>excess`, `<prefix>in_bat`, `<prefix>feed_in`,
        `<prefix>self_use`, `<prefix>self_share`.

    Example:
        ```python
        df = add_energy_flows_multi(
            df,
            sources=[
                {"prod_col": "pv_power", "prod_is_positive": False, "prefix": "pv_"},
                {"prod_col": "wind_power", "prod_is_positive": True, "prefix": "wind_"},
            ],
            cascade_consumption=True,
            cascade_battery=True,
        )
        ```

        # Resulting columns:
        #   pv_prod, pv_excess, pv_in_bat, pv_feed_in, pv_self_use, pv_self_share
        #   wind_prod, wind_excess, wind_in_bat, wind_feed_in, wind_self_use, wind_self_share
    """
    out = df.copy()

    # Start with global consumption/battery; allow None to mean "no column"
    remaining_consumption = _get_series(out, consumption_col)
    remaining_battery = _get_series(out, battery_charge_col)

    tmp_cols_to_drop = []

    for idx, s in enumerate(sources):
        prefix = s.get("prefix", "")
        names = {
            "self_use": (s.get("out_cols", {}) or {}).get(
                "self_use", f"{prefix}self_use"
            ),
            "in_bat": (s.get("out_cols", {}) or {}).get("in_bat", f"{prefix}in_bat"),
        }

        # Prepare per-iteration consumption/battery inputs
        cons_col_for_source = s.get("consumption_col", consumption_col)
        bat_col_for_source = s.get("battery_charge_col", battery_charge_col)

        if cascade_consumption:
            cons_col_for_source = f"{tmp_prefix}cons_{idx}"
            out[cons_col_for_source] = remaining_consumption
            tmp_cols_to_drop.append(cons_col_for_source)

        if cascade_battery:
            bat_col_for_source = f"{tmp_prefix}bat_{idx}"
            out[bat_col_for_source] = remaining_battery
            tmp_cols_to_drop.append(bat_col_for_source)

        # Compute and assign this source's flows
        out = _add_energy_flows(
            out,
            prod_col=s["prod_col"],
            prod_is_positive=s["prod_is_positive"],
            consumption_col=cons_col_for_source,
            battery_charge_col=bat_col_for_source,
            out_prefix=prefix,
            out_cols=s.get("out_cols"),
            assign=True,
        )

        # Update remaining consumption/battery for next sources
        if cascade_consumption and names["self_use"] in out.columns:
            remaining_consumption = (
                remaining_consumption - out[names["self_use"]]
            ).clip(lower=0)

        if cascade_battery and names["in_bat"] in out.columns:
            remaining_battery = (remaining_battery - out[names["in_bat"]]).clip(lower=0)

    # Clean up temp inputs
    if tmp_cols_to_drop:
        out = out.drop(columns=[c for c in tmp_cols_to_drop if c in out.columns])

    return out
