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
  - `create_battery_usecase_df`: Builds the standardized battery-usecase table
    used by the reporting plot helpers.

Usage:
-----
Use create_energy_report_df() inside reporting pipelines or notebooks to
transform raw microgrid exports into localized, labeled, and analysis-ready
tables for KPIs, dashboards, and stakeholder reporting.
"""

import pandas as pd
from frequenz.gridpool import MicrogridConfig

from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper
from frequenz.lib.notebooks.reporting.utils.helpers import (
    AggregatedComponentConfig,
    add_energy_flows,
    convert_timezone,
    fill_aggregated_component_columns,
    get_energy_report_columns,
    label_component_columns,
)


def create_battery_usecase_df(
    energy_report_df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    grid_consumption_col: str = "grid_consumption",
    battery_col: str = "battery_power_flow",
    pv_col: str | None = "pv_asset_production",
) -> pd.DataFrame:
    """Create a standardized battery-usecase DataFrame.

    Selects the battery-usecase input columns from the source DataFrame, renames
    them to the standardized reporting schema, derives grid consumption without
    battery support, computes reference peak lines, and splits battery power
    flow into charging and discharging series.

    Args:
        energy_report_df: Reporting DataFrame containing the source columns for
            timestamp, grid consumption, and battery power flow.
        timestamp_col: Column name in `energy_report_df` containing the
            timestamps.
        grid_consumption_col: Column name in `energy_report_df` containing
            grid consumption with battery support.
        battery_col: Column name in `energy_report_df` containing
            battery power flow.
        pv_col: Optional column name in `energy_report_df` containing
            PV production to preserve in the standardized output when present.

    Returns:
        The battery-usecase DataFrame with derived helper columns for plotting
        and analysis.

    Raises:
        KeyError: If required columns are missing from the input DataFrame.
    """
    required_cols = [timestamp_col, grid_consumption_col, battery_col]
    missing_cols = [col for col in required_cols if col not in energy_report_df.columns]
    if missing_cols:
        raise KeyError(
            "Missing required columns in energy_report_df: "
            + ", ".join(sorted(missing_cols))
        )

    selected_cols = list(required_cols)
    rename_map = {
        timestamp_col: "timestamp",
        grid_consumption_col: "grid_consumption",
        battery_col: "battery_power_flow",
    }
    if pv_col and pv_col in energy_report_df.columns:
        selected_cols.append(pv_col)
        rename_map[pv_col] = "pv"

    battery_usecase_df = energy_report_df[selected_cols].rename(columns=rename_map)
    battery_usecase_df["grid_consumption_without_battery"] = (
        battery_usecase_df["grid_consumption"]
        + battery_usecase_df["battery_power_flow"]
    )

    battery_usecase_df["peak_before_optimization"] = battery_usecase_df[
        "grid_consumption_without_battery"
    ].max()
    battery_usecase_df["peak_after_optimization"] = battery_usecase_df[
        "grid_consumption"
    ].max()
    battery_usecase_df["battery_discharge"] = battery_usecase_df[
        "battery_power_flow"
    ].clip(lower=0)
    battery_usecase_df["battery_charge"] = battery_usecase_df[
        "battery_power_flow"
    ].clip(upper=0)

    return battery_usecase_df


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
    )

    # Determine relevant columns based on component types
    energy_report_df_cols = get_energy_report_columns(
        component_types, single_components
    )

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
