# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Microgrid Reporting DataFrame Construction.

This module constructs normalized energy-report DataFrames from
raw microgrid telemetry by harmonizing timestamps and column naming,
enriching PV flows, adding grid KPIs, and surfacing component-specific
metrics used downstream for dashboards.

Functions:
------------
- Energy Report Data Fetching
  - `fetch_energy_report_data`: Fetches power data directly, or replaces net
    grid energy with consumed and delivered energy columns.
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

from datetime import datetime, timedelta

import pandas as pd

from frequenz.data.microgrid import MicrogridConfig, MicrogridData
from frequenz.lib.notebooks.reporting.metrics.reporting_metrics import GridDataSource
from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper
from frequenz.lib.notebooks.reporting.utils.helpers import (
    AggregatedComponentConfig,
    add_energy_flows,
    convert_timezone,
    fill_aggregated_component_columns,
    get_energy_report_columns,
    label_component_columns,
)


async def fetch_energy_report_data(  # pylint: disable=too-many-arguments
    mdata: MicrogridData,
    *,
    microgrid_id: int | str,
    component_types: tuple[str, ...],
    start: datetime,
    end: datetime,
    resampling_period: timedelta = timedelta(seconds=10),
    keep_components: bool = False,
    grid_data_source: GridDataSource = "power",
) -> pd.DataFrame | None:
    """Fetch the source data used to construct an energy report.

    Power data contains one signed ``grid`` column and can be returned as-is.
    Energy data needs separate import and export measurements, so the net grid
    column is replaced with directional ``grid_energy_consumed`` and
    ``grid_energy_delivered`` columns.

    Args:
        mdata: Microgrid data client used for the reporting queries.
        microgrid_id: ID of the microgrid to query.
        component_types: Component types to include in the report data.
        start: Start of the reporting interval.
        end: End of the reporting interval.
        resampling_period: Resolution of the returned data.
        keep_components: Whether to include individual component columns.
        grid_data_source: Fetch signed active power or active-energy deltas.

    Returns:
        The fetched report data, or ``None`` when the main query has no data.

    Raises:
        ValueError: If ``grid_data_source`` is unknown, or a directional energy
            query does not return a ``grid`` column.
    """
    if grid_data_source == "power":
        return await mdata.ac_active_power(
            microgrid_id=microgrid_id,
            component_types=component_types,
            start=start,
            end=end,
            resampling_period=resampling_period,
            keep_components=keep_components,
            unit="kW",
        )
    if grid_data_source != "energy":
        raise ValueError(f"Unknown grid data source: {grid_data_source}")

    energy_df = await mdata.ac_active_energy(
        microgrid_id=microgrid_id,
        component_types=component_types,
        start=start,
        end=end,
        resampling_period=resampling_period,
        keep_components=keep_components,
        unit="kWh",
    )
    if energy_df is None:
        return None

    consumed_grid_df = await mdata.ac_active_energy(
        microgrid_id=microgrid_id,
        component_types=("grid",),
        start=start,
        end=end,
        resampling_period=resampling_period,
        keep_components=keep_components,
        unit="kWh",
        direction="consumed",
    )
    delivered_grid_df = await mdata.ac_active_energy(
        microgrid_id=microgrid_id,
        component_types=("grid",),
        start=start,
        end=end,
        resampling_period=resampling_period,
        keep_components=keep_components,
        unit="kWh",
        direction="delivered",
    )

    directional_frames = {
        "consumed": consumed_grid_df,
        "delivered": delivered_grid_df,
    }
    for direction, directional_df in directional_frames.items():
        if directional_df is None or "grid" not in directional_df.columns:
            raise ValueError(
                f"The {direction} active-energy query did not return a grid column."
            )

    result = energy_df.drop(
        columns=["grid"],
        errors="ignore",
    )
    assert consumed_grid_df is not None
    assert delivered_grid_df is not None
    result["grid_energy_consumed"] = consumed_grid_df["grid"]
    result["grid_energy_delivered"] = delivered_grid_df["grid"]
    return result


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
    grid_data_source: GridDataSource = "power",
    resolution: timedelta | None = None,
) -> pd.DataFrame:
    """Create a normalized Energy Report DataFrame with selected columns.

    Makes a copy of the input, converts the timestamp column to the configured
    timezone, renames standard columns to unified names, adds the net import
    column, renames numeric component IDs to labeled names, and returns a
    reduced DataFrame containing only relevant columns.

    Args:
        df: Raw input table containing energy data. Power mode expects a signed
            grid column. Energy mode expects separate grid_energy_consumed and
            grid_energy_delivered columns.
        component_types: Component types to include in the Energy Report DataFrame
                (e.g., battery, pv).
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
            notebook code after awaiting get_meter_display_names().
        battery_soc_df: Optional SOC data returned by MicrogridData.soc().
            When battery is among component_types, its battery column is added
            as battery_soc_pct.
        grid_data_source: "power" to use signed grid power or "energy" to use
            directional energy deltas.
        resolution: Reporting period duration required when grid_data_source is
            "energy". Energy deltas are converted to average power so
            the canonical report columns retain their existing units.

    Returns:
        The Energy Report DataFrame with standardized and selected columns.

    Notes:
        Component IDs are renamed to labeled names via label_component_columns().
    """
    energy_report_df = df.copy()

    # Only reset index if it's a datetime or period index and 'timestamp' column is missing
    if isinstance(energy_report_df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if "timestamp" not in energy_report_df.columns:
            energy_report_df = energy_report_df.reset_index(names="timestamp")

    battery_soc_columns = [
        canonical
        for canonical in mapper.canonicals
        if canonical == "battery_soc_pct" or canonical.startswith("battery_soc_")
    ]
    if "battery" in component_types and battery_soc_df is not None:
        energy_report_df = _add_battery_soc_column(
            energy_report_df,
            battery_soc_df,
            mapper,
            battery_soc_columns,
        )

    # Add Energy flow columns
    energy_report_df = add_energy_flows(
        energy_report_df,
        production_cols=["pv", "chp", "wind"],
        consumption_cols=["consumption"],
        grid_energy_consumed_cols=["grid_energy_consumed"],
        grid_energy_delivered_cols=["grid_energy_delivered"],
        battery_cols=["battery"],
        grid_data_source=grid_data_source,
        resolution=resolution,
    )
    raw_grid_column = mapper.canonical_to_raw["grid_consumption"]
    if grid_data_source == "power" and raw_grid_column != "grid_consumption":
        energy_report_df = energy_report_df.drop(
            columns=[raw_grid_column], errors="ignore"
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
    if "battery" in component_types and battery_soc_df is not None:
        energy_report_df_cols.extend(
            col
            for col in battery_soc_columns
            if col in energy_report_df.columns and col not in energy_report_df_cols
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


def _add_battery_soc_column(
    energy_report_df: pd.DataFrame,
    battery_soc_df: pd.DataFrame | None,
    mapper: ColumnMapper,
    battery_soc_columns: list[str],
) -> pd.DataFrame:
    """Add schema-defined battery SOC fields from fetched SOC data."""
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
    soc_raw_column = mapper.canonical_to_raw["battery_soc_pct"]
    soc_df = mapper.to_canonical(soc_df.rename(columns={"battery": soc_raw_column}))
    available_soc_cols = [col for col in battery_soc_columns if col in soc_df.columns]
    soc_df = soc_df[["timestamp", *available_soc_cols]]
    return result.merge(soc_df, on="timestamp", how="left")
