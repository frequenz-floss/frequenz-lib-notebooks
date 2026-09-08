# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Microgrid Reporting DataFrame Construction.

This module constructs normalized energy-report DataFrames from
raw microgrid telemetry by harmonizing timestamps and column naming,
enriching PV flows, adding grid KPIs, and surfacing component-specific
metrics used downstream for dashboards.

Functions:
------------
- Energy Report DataFrame Construction
  - `create_energy_report_df`: Builds a normalized energy report table with
    unified naming, timezone conversion, grid import calculation, and
    component renaming based on a MicrogridConfig.

Usage:
-----
Use create_energy_report_df() inside reporting pipelines or notebooks to
transform raw microgrid exports into localized, labeled, and analysis-ready
tables for KPIs, dashboards, and stakeholder reporting.
"""

import pandas as pd

from frequenz.data.microgrid import MicrogridConfig
from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper
from frequenz.lib.notebooks.reporting.utils.helpers import (
    AggregatedComponentConfig,
    add_energy_flows,
    convert_timezone,
    fill_aggregated_component_columns,
    get_energy_report_columns,
    label_component_columns,
)


# pylint: disable=too-many-arguments, too-many-locals
def create_energy_report_df(
    df: pd.DataFrame,
    component_types: list[str],
    mcfg: MicrogridConfig,
    mapper: ColumnMapper,
    *,
    tz_name: str = "Europe/Berlin",
    assume_tz: str = "UTC",
    fill_missing_values: bool = True,
    aggregated_component_config: AggregatedComponentConfig | None = None,
    component_display_names: dict[str, str] | None = None,
    battery_soc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a normalized Energy Report DataFrame with selected columns.

    Makes a copy of the input, converts the timestamp column to the configured
    timezone, renames standard columns to unified names, adds the net import
    column, renames numeric component IDs to labeled names, and returns a
    reduced DataFrame containing only relevant columns.

    Args:
        df: Raw input table containing energy data.
        component_types: Component types to include in the Energy Report DataFrame
                (e.g., ``battery``, ``pv``).
        mcfg: Configuration object used to resolve component IDs.
        mapper: Column Mapper object to standardize the column names.
        tz_name: Target timezone name for timestamp conversion (default: "Europe/Berlin").
        assume_tz: Timezone to assume for naive datetimes before conversion (default: "UTC").
        fill_missing_values: Whether to fill missing aggregate component columns
                from per-component sums (default: True).
        aggregated_component_config: Optional mapping of component types to aggregated
            column metadata used when filling missing aggregates. Defaults to the shared
            `DEFAULT_AGGREGATED_COMPONENT_CONFIG`.
        component_display_names: Optional mapping from numeric component IDs to
            display names fetched from the Assets API, typically passed in from
            notebook code after awaiting ``get_meter_display_names()``.
        battery_soc_df: Optional SOC data returned by ``MicrogridData.soc()``.
            When battery is present and ``mcfg`` has a ``BATTERY_SOC_PCT``
            formula, its ``battery`` column is added as ``battery_soc_pct``.

    Returns:
        The Energy Report DataFrame with standardized and selected columns.

    Notes:
        Component IDs are renamed to labeled names via ``label_component_columns()``.
    """
    energy_report_df = df.copy()

    # Only reset index if it's a datetime or period index and 'timestamp' column is missing
    if isinstance(energy_report_df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if "timestamp" not in energy_report_df.columns:
            energy_report_df = energy_report_df.reset_index(names="timestamp")

    include_battery_soc = _has_battery_soc_formula(component_types, mcfg)
    if include_battery_soc:
        energy_report_df = _add_battery_soc_column(energy_report_df, battery_soc_df)

    # Add Energy flow columns
    energy_report_df = add_energy_flows(
        energy_report_df,
        production_cols=["pv", "chp", "wind"],
        consumption_cols=["consumption"],
        grid_cols=["grid"],
        battery_cols=["battery"],
    )

    # Standardize column names (from raw to canonical)
    energy_report_df = mapper.to_canonical(energy_report_df)

    # Convert timestamp to datetime if not already
    energy_report_df["timestamp"] = pd.to_datetime(
        energy_report_df["timestamp"], errors="coerce", utc=True
    )

    # Convert timezone
    energy_report_df["timestamp"] = convert_timezone(
        energy_report_df["timestamp"],
        target_tz=tz_name,
        assume_tz=assume_tz,
    )

    # Helper to rename numeric component IDs to labeled names like PV #250, Battery #219
    # (casing matches output format)
    energy_report_df, single_components = label_component_columns(
        energy_report_df,
        mcfg,
        column_battery="battery",
        column_pv="pv",
        column_chp="chp",
        column_ev="ev",
        column_wind="wind",
        component_display_names=component_display_names,
    )

    # Determine relevant columns based on component types
    energy_report_df_cols = get_energy_report_columns(
        component_types, single_components
    )
    if include_battery_soc and "battery_soc_pct" in energy_report_df.columns:
        energy_report_df_cols.append("battery_soc_pct")

    # Select only the relevant columns
    energy_report_df = energy_report_df[energy_report_df_cols]

    if fill_missing_values:
        # Fill in missing aggregate component columns from per-component sums
        energy_report_df = fill_aggregated_component_columns(
            energy_report_df,
            component_types,
            aggregated_component_config,
        )

    return energy_report_df


def _has_battery_soc_formula(component_types: list[str], mcfg: MicrogridConfig) -> bool:
    """Return whether the report should include battery SOC."""
    if "battery" not in component_types:
        return False
    try:
        mcfg.formula("battery", "BATTERY_SOC_PCT")
    except (AttributeError, ValueError):
        return False
    return True


def _add_battery_soc_column(
    energy_report_df: pd.DataFrame,
    battery_soc_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add a canonical battery SOC column from fetched or pre-merged SOC data."""
    if battery_soc_df is None:
        return energy_report_df

    if "battery" not in battery_soc_df.columns:
        raise KeyError("battery_soc_df must contain a 'battery' column.")

    result = energy_report_df.copy()
    soc_df = battery_soc_df.copy()
    if isinstance(soc_df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        soc_df = soc_df.reset_index(names="timestamp")
    if "timestamp" not in soc_df.columns:
        raise KeyError("battery_soc_df must contain a 'timestamp' column or index.")

    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    soc_df["timestamp"] = pd.to_datetime(soc_df["timestamp"], errors="coerce", utc=True)
    soc_df = soc_df[["timestamp", "battery"]].rename(
        columns={"battery": "battery_soc_pct"}
    )
    return result.merge(soc_df, on="timestamp", how="left")
