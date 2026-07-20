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

import warnings

import pandas as pd
from frequenz.client.base.exception import ApiClientError
from frequenz.gridpool import MicrogridConfig

from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper
from frequenz.lib.notebooks.reporting.utils.component_metadata import (
    ComponentMetadata,
    EnergyReport,
)
from frequenz.lib.notebooks.reporting.utils.helpers import (
    AggregatedComponentConfig,
    add_energy_flows,
    convert_timezone,
    fill_aggregated_component_columns,
    get_component_ids,
    get_energy_report_columns,
    get_meter_display_names,
)


def _resolve_component_metadata(
    energy_report_df: pd.DataFrame,
    component_types: list[str],
    mcfg: MicrogridConfig,
    component_display_names: dict[str, str] | None,
    *,
    include_display_names: bool,
) -> tuple[ComponentMetadata, list[str]]:
    """Resolve component display metadata and canonical component columns."""
    resolved_display_names = component_display_names if include_display_names else {}
    if include_display_names and resolved_display_names is None:
        microgrid_id = getattr(getattr(mcfg, "meta", None), "microgrid_id", None)
        if microgrid_id is not None:
            try:
                resolved_display_names = get_meter_display_names(int(microgrid_id))
            except ValueError:
                resolved_display_names = None
            except ApiClientError as exc:  # pragma: no cover - defensive fallback
                warnings.warn(
                    f"Could not fetch meter display names from the Assets API: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                resolved_display_names = None

    component_ids_by_type = {
        component_type: [
            component_id
            for component_id in get_component_ids(mcfg, component_type)
            if component_id in energy_report_df.columns
        ]
        for component_type in component_types
    }
    single_components = [
        component_id
        for component_type in component_types
        for component_id in component_ids_by_type.get(component_type, [])
    ]

    return (
        ComponentMetadata(
            display_names=resolved_display_names or {},
            ids_by_type=component_ids_by_type,
        ),
        single_components,
    )


# pylint: disable=too-many-arguments
def _build_energy_report_dataframe(
    df: pd.DataFrame,
    component_types: list[str],
    mapper: ColumnMapper,
    *,
    tz_name: str,
    assume_tz: str,
    fill_missing_values: bool,
    aggregated_component_config: AggregatedComponentConfig | None,
    component_metadata: ComponentMetadata,
) -> pd.DataFrame:
    """Apply the dataframe transformation pipeline for energy reporting."""
    energy_report_df = df.copy()

    if isinstance(energy_report_df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if "timestamp" not in energy_report_df.columns:
            energy_report_df = energy_report_df.reset_index(names="timestamp")

    energy_report_df = add_energy_flows(
        energy_report_df,
        production_cols=["pv", "chp", "wind"],
        consumption_cols=["consumption"],
        grid_cols=["grid"],
        battery_cols=["battery"],
    )
    energy_report_df = mapper.to_canonical(energy_report_df)
    energy_report_df["timestamp"] = pd.to_datetime(
        energy_report_df["timestamp"], errors="coerce", utc=True
    )
    energy_report_df["timestamp"] = convert_timezone(
        energy_report_df["timestamp"],
        target_tz=tz_name,
        assume_tz=assume_tz,
    )

    energy_report_df_cols = get_energy_report_columns(
        component_types,
        [
            component_id
            for component_ids in component_metadata.ids_by_type.values()
            for component_id in component_ids
        ],
    )
    energy_report_df = energy_report_df[energy_report_df_cols]

    if fill_missing_values:
        energy_report_df = fill_aggregated_component_columns(
            energy_report_df,
            component_types,
            aggregated_component_config,
            component_columns_by_type=component_metadata.ids_by_type,
        )

    return energy_report_df


# pylint: disable=too-many-arguments
def build_energy_report(
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
    include_component_metadata: bool = True,
) -> EnergyReport:
    """Build the normalized energy report and optional component metadata."""
    metadata, _ = _resolve_component_metadata(
        df,
        component_types,
        mcfg,
        component_display_names,
        include_display_names=include_component_metadata,
    )
    energy_report_df = _build_energy_report_dataframe(
        df,
        component_types,
        mapper,
        tz_name=tz_name,
        assume_tz=assume_tz,
        fill_missing_values=fill_missing_values,
        aggregated_component_config=aggregated_component_config,
        component_metadata=metadata,
    )

    return EnergyReport(df=energy_report_df, metadata=metadata)


# pylint: disable=too-many-arguments, too-many-locals
def create_battery_usecase_df(
    energy_report_df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    consumption_col: str = "mid_consumption",
    grid_consumption_col: str = "grid_consumption",
    battery_col: str = "battery_power_flow",
    pv_col: str | None = "pv_asset_production",
    chp_col: str | None = "chp_asset_production",
    wind_col: str | None = "wind_asset_production",
) -> pd.DataFrame:
    """Create a standardized battery-usecase DataFrame.

    Selects the battery-usecase input columns from the source DataFrame, renames
    them to the standardized reporting schema, keeps the site consumption
    baseline used for plotting, computes reference peak lines, and splits
    battery power flow into charging and discharging series.

    Args:
        energy_report_df: Reporting DataFrame containing the source columns for
            timestamp, consumption, grid consumption, and battery power flow.
        timestamp_col: Column name in `energy_report_df` containing the
            timestamps.
        consumption_col: Column name in `energy_report_df` containing
            site consumption.
        grid_consumption_col: Column name in `energy_report_df` containing
            grid consumption with battery support.
        battery_col: Column name in `energy_report_df` containing
            battery power flow.
        pv_col: Optional column name in `energy_report_df` containing
            PV production to preserve in the standardized output when present.
        chp_col: Optional column name in `energy_report_df` containing
            CHP production to preserve in the standardized output when present.
        wind_col: Optional column name in `energy_report_df` containing
            wind production to preserve in the standardized output when present.

    Returns:
        The battery-usecase DataFrame with derived helper columns for plotting
        and analysis.

    Raises:
        KeyError: If required columns are missing from the input DataFrame.
    """
    required_cols = [timestamp_col, consumption_col, grid_consumption_col, battery_col]
    missing_cols = [col for col in required_cols if col not in energy_report_df.columns]
    if missing_cols:
        raise KeyError(
            "Missing required columns in energy_report_df: "
            + ", ".join(sorted(missing_cols))
        )

    selected_cols = list(required_cols)
    rename_map = {
        timestamp_col: "timestamp",
        consumption_col: "consumption",
        grid_consumption_col: "grid_consumption",
        battery_col: "battery_power_flow",
    }
    optional_production_columns = {
        pv_col: "pv",
        chp_col: "chp",
        wind_col: "wind",
    }
    for source_col, target_col in optional_production_columns.items():
        if source_col and source_col in energy_report_df.columns:
            selected_cols.append(source_col)
            rename_map[source_col] = target_col

    battery_usecase_df = energy_report_df[selected_cols].rename(columns=rename_map)
    battery_usecase_df["peak_before_optimization"] = battery_usecase_df[
        "consumption"
    ].max()
    battery_usecase_df["peak_after_optimization"] = battery_usecase_df[
        "grid_consumption"
    ].max()
    battery_usecase_df["battery_charge"] = battery_usecase_df[
        "battery_power_flow"
    ].clip(lower=0)
    battery_usecase_df["battery_discharge"] = battery_usecase_df[
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
    component_display_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create a normalized Energy Report DataFrame with selected columns.

    This is a dataframe-only compatibility wrapper around ``build_energy_report()``.

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
            display names that are surfaced by ``build_energy_report()`` when
            callers need metadata as well.

    Returns:
        The Energy Report DataFrame.

    Notes:
        Use ``build_energy_report()`` when callers also need component metadata.
    """
    return build_energy_report(
        df,
        component_types,
        mcfg,
        mapper,
        tz_name=tz_name,
        assume_tz=assume_tz,
        fill_missing_values=fill_missing_values,
        aggregated_component_config=aggregated_component_config,
        component_display_names=component_display_names,
        include_component_metadata=False,
    ).df
