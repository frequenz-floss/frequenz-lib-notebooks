# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Microgrid Reporting DataFrame Construction.

This module constructs normalized energy-report DataFrames from
raw microgrid telemetry by harmonizing timestamps and column naming,
enriching PV flows, adding grid KPIs, and surfacing component-specific
metrics used downstream for dashboards.

Key Features
------------
- Energy Report DataFrame Construction
  - :func:`create_energy_report_df`: Builds a normalized energy report table with
    unified naming, timezone conversion, grid import calculation, and
    component renaming based on a MicrogridConfig.

Usage
-----
Use create_energy_report_dfs() inside reporting pipelines or notebooks to
transform raw microgrid exports into localized, labeled, and analysis-ready
tables for KPIs, dashboards, and stakeholder reporting.
"""


from typing import List

import pandas as pd

from frequenz.data.microgrid.config import MicrogridConfig
from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper
from frequenz.lib.notebooks.reporting.utils.helpers import (
    _add_pv_energy_flows,
    _convert_timezone,
    add_net_grid_import,
    get_energy_report_columns,
    label_component_columns,
)


# pylint: disable=too-many-arguments, disable=too-many-locals
def create_energy_report_dfs(
    df: pd.DataFrame,
    component_types: List[str],
    mcfg: MicrogridConfig,
    mapper: ColumnMapper,
    *,
    tz_name: str = "Europe/Berlin",
    assume_tz: str = "UTC",
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

    # Add PV energy flow columns
    energy_report_df = _add_pv_energy_flows(energy_report_df)

    # Standardize column names (from raw to canonical)
    energy_report_df = mapper.to_canonical(energy_report_df)

    # Convert timezone
    energy_report_df = _convert_timezone(
        energy_report_df,
        column_timestamp="timestamp",
        target_tz=tz_name,
        assume_tz=assume_tz,
    )

    # Add grid consumption column
    energy_report_df = add_net_grid_import(
        energy_report_df,
        column_grid="grid",
        column_net_import="net_import",
    )

    # Helper to rename numeric component IDs to labeled names like PV #250, Battery #219
    energy_report_df, single_components = label_component_columns(
        energy_report_df,
        mcfg,
        column_battery="battery",
        column_pv="pv",
    )

    energy_report_df_cols = get_energy_report_columns(
        component_types, single_components
    )

    # Select only the relevant columns
    energy_report_df = energy_report_df[energy_report_df_cols]
    return energy_report_df
