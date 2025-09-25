# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH
"""Helper function for Microgrid Data Processing Utilities.

This module provides utility functions for preprocessing and analyzing microgrid
data represented in pandas DataFrames. It standardizes column names, handles
timezone conversions, computes grid imports, derives photovoltaic (PV) energy flows,
and renames component-specific columns based on a MicrogridConfig.

Key Features
------------
- Timezone Conversion
  Ensures all timestamps are consistently localized
  (default: UTC → Europe/Berlin).

- Grid Data Processing
  Extracts net grid import by filtering positive values
  from grid connection signals.

- PV Energy Flow Calculations
  Derives PV production, excess, self-consumption, battery charging, and
  grid feed-in metrics, including PV self-consumption share.

- Component Renaming
  Maps numeric string component IDs to human-readable labels
  (e.g., "Battery #14", "PV #7") using the provided MicrogridConfig.

- Reporting Column Assembly
  Builds the column sets required for downstream energy reports
  based on the available component types.

Usage
-----
These functions serve as building blocks for energy reporting, data pipelines,
and dashboards that analyze microgrid performance, particularly in hybrid systems
with PV, batteries, and grid interactions.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from frequenz.data.microgrid.config import MicrogridConfig


def load_config(path: str) -> Dict[str, Any]:
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
        raise TypeError("YAML root must be a mapping (dict).")

    return data


def _fmt_de(x: float) -> str:
    """Format a number using German-style decimal and thousands separators.

    The function formats the number with two decimal places, using a comma
    as the decimal separator and a dot as the thousands separator.

    Args:
        x: The number to format.

    Returns:
        The formatted string with German number formatting applied.

    Example:
        >>> _fmt_de(12345.6789)
        '12.345,68'
    """
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _convert_timezone(
    df: pd.DataFrame,
    column_timestamp: str,
    target_tz: str = "Europe/Berlin",
    assume_tz: str = "UTC",
) -> pd.DataFrame:
    """Convert a datetime column in a DataFrame to a target timezone.

    If the column contains timezone-naive datetimes, they are first localized to
    ``assume_tz`` before being converted to ``target_tz``.

    Args:
        df: Input DataFrame containing the datetime column.
        column_timestamp: Name of the datetime column in ``df`` to convert.
        target_tz: Timezone name to convert the column to.
            Defaults to ``"Europe/Berlin"``.
        assume_tz: Timezone to assume for naive datetimes.
            Defaults to ``"UTC"``.

    Returns:
        pd.DataFrame: A copy of the DataFrame with the converted datetime column.

    Raises:
        ValueError: If ``column_timestamp`` is not present in ``df``.
    """
    if column_timestamp not in df:
        raise ValueError(f"{column_timestamp} column not in df")

    ts = df[column_timestamp]

    if ts.dt.tz is None:
        # Assume naïve datetimes are in `assume_tz`
        ts = ts.dt.tz_localize(assume_tz)

    df[column_timestamp] = ts.dt.tz_convert(target_tz)
    return df


def add_net_grid_import(
    df: pd.DataFrame,
    column_grid: str,
    column_net_import: str,
) -> pd.DataFrame:
    """Calculate grid consumption and add it as ``column_net_import``.

    Grid consumption is defined as the positive part of ``column_grid``.
    Negative values are replaced with 0.

    Args:
        df: Input DataFrame containing the grid data.
        column_grid: Name of the column in ``df`` that contains grid values.
        column_net_import: Name of the output column to store the computed
            net import values.

    Returns:
        pd.DataFrame: The DataFrame with a new or updated ``column_net_import`` column.

    Raises:
        ValueError: If ``column_grid`` is not present in ``df``.
    """
    if column_grid not in df:
        raise ValueError(f"{column_grid} column not in df")

    df[column_net_import] = df[column_grid].apply(lambda x: x if x > 0 else 0)
    return df


# pylint: disable=too-many-arguments, too-many-positional-arguments
def label_component_columns(
    df: pd.DataFrame,
    mcfg: MicrogridConfig,
    column_battery: str = "battery",
    column_pv: str = "pv",
    column_chp: str = "chp",
    column_ev: str = "ev",
) -> Tuple[pd.DataFrame, List[str]]:
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
        column_ev: Key name for EV component type
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

    rename: Dict[str, str] = {}
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

    return df.rename(columns=rename), list(rename.values())


def _add_pv_energy_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Add PV-related energy flow columns to ``df`` if PV data is present.

    Derives photovoltaic (PV) energy-flow metrics from existing columns. If no PV
    signal is present (i.e., the negative PV column is missing or all zeros), the
    DataFrame is returned unchanged.

    Args:
      df: Input DataFrame. If present, uses columns ``pv_neg``,
        ``consumption``, and ``COLUMN_BATTERY_POS``. Missing columns are
        treated as zeros.

    Returns:
      The DataFrame with added PV flow columns (or unchanged if no PV signal).

    Notes:
      Newly created/updated columns:
        - ``COLUMN_PV_PROD``: PV production as a positive series (negated/clipped from
          ``pv_neg``).
        - ``COLUMN_PV_EXCESS``: Excess PV after subtracting household consumption.
        - ``COLUMN_PV_BAT``: Portion of PV excess routed into the battery (bounded by
          battery charge).
        - ``COLUMN_PV_FEEDIN``: PV fed into the grid after battery charging.
        - ``COLUMN_PV_SELF``: Self-consumed PV (production minus excess).
        - ``COLUMN_PV_SHARE``: Share of consumption covered by self-consumed PV (NaN
          when consumption is 0).
    """
    # Safe inputs (0 if missing)
    df_with_pv_flows = df.copy()
    zeros = pd.Series(0, index=df_with_pv_flows.index)
    pv_neg = df_with_pv_flows.get("pv_neg", zeros)
    consumption = df_with_pv_flows.get("consumption", zeros)
    battery_pos = df_with_pv_flows.get("battery_pos", zeros)

    # Only compute PV features if there is any PV signal
    has_pv = isinstance(pv_neg, pd.Series) and (pv_neg != 0).any()
    if not has_pv:
        return df_with_pv_flows

    df_with_pv_flows["pv_prod"] = (-pv_neg).clip(lower=0)
    df_with_pv_flows["pv_excess"] = (df_with_pv_flows["pv_prod"] - consumption).clip(
        lower=0
    )

    # This naturally becomes 0 when there's no battery_pos column
    df_with_pv_flows["pv_bat"] = pd.concat(
        [df_with_pv_flows["pv_excess"], battery_pos], axis=1
    ).min(axis=1)

    df_with_pv_flows["pv_feedin"] = (
        df_with_pv_flows["pv_excess"] - df_with_pv_flows["pv_bat"]
    )
    df_with_pv_flows["pv_self"] = (
        df_with_pv_flows["pv_prod"] - df_with_pv_flows["pv_excess"]
    ).clip(lower=0)

    denom = consumption.replace(0, pd.NA)
    df_with_pv_flows["pv_share"] = df_with_pv_flows["pv_self"] / denom

    return df_with_pv_flows


def get_energy_report_columns(
    component_types: List[str], single_components: List[str]
) -> List[str]:
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
        "grid",
        "net_import",
        "net_consumption",
    ] + single_components

    # Map component types to the columns they enable
    component_column_map = {
        "battery": ["battery_throughput"],
        "pv": [
            "pv_throughput",
            "pv_prod",
            "pv_self",
            "pv_feedin",
        ],
    }

    # Define columns that require both PV and Battery
    pv_battery_cols = [
        "pv_in_bat",
        "pv_share",
    ]

    # Add component-specific columns
    for component, columns in component_column_map.items():
        if component in component_types:
            energy_report_df_cols.extend(columns)

    # Add combined PV + Battery columns
    if "pv" in component_types and "battery" in component_types:
        energy_report_df_cols.extend(pv_battery_cols)

    return energy_report_df_cols
