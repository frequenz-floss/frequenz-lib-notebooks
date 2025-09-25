# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Microgrid Reporting and Analysis Utilities for the notebook.

This module provides helpers to compute energy summaries, normalize per-component
timeseries to tidy (long) tables, and derive high-level KPIs.

Key features
-------------
1) compute_energy_summary(df, resolution) -> DataFrame
   Aggregates PV and grid energy over the time step and returns a compact table with:
     - "Energy Source"  (PV / Grid)
     - "Energy [kWh]"   (sum over period)
     - "Energy %"       (share of total)
     - "Power [kW]"     (avg power = energy / hours)
   Looks for "net_import" (grid) and "pv_self" (self-consumed PV). Missing series
   are treated as zero; rows are ordered PV first (if present), then Grid.

2) display_pv_energy(energy_report_df, resolution, label_contains="PV #", …) -> dict
   Prints per-component PV energies in kWh (supports German formatting) and returns
   a dict of {component_label: kWh}. Optionally prints and returns "PV Total".
   Uses the convention that PV power is negative and multiplies by -1.

3) build_pv_analysis_df(energy_report_df, pv_filter) -> DataFrame
   Creates a tidy, long-form PV table by melting selected "PV #N" columns into:
     - "timestamp"
     - optional "grid" (evenly divided across selected PVs if present)
     - "PV" (component label without the "PV " prefix, e.g., "#1")
     - "pv_feedin" (positive production after sign flip)
   `pv_filter` accepts explicit IDs like ["#1", "#3"] or ["all"] (case-insensitive).

4) build_component_analysis(energy_report_df, selection_filter,
   component_label, value_col_name) -> DataFrame
   Generic version of (3) for any component family, e.g. ("Battery", "battery").
   Returns long form with columns: "timestamp", <component_label>, <value_col_name>.

5) build_overview_df(energy_report_df, component_types) -> DataFrame
   Returns a trimmed view with essential base columns
     ["timestamp", "net_import", "net_consumption"]
   plus optional groups if requested (e.g., "pv" -> ["pv_prod", "pv_feedin"],
   "battery" -> ["battery_throughput"]). Missing columns are safely ignored.

6) load_microgrid_configs(directory="toml_directory/") -> (dict, list[int])
   Loads all *.toml files via `MicrogridConfig.load_configs(...)`, returning:
     - configs: dict[str, MicrogridConfig]
     - available_microgrids: sorted list of int IDs

7) aggregate_pv_metrics(energy_report_df, resolution,
   grid_consumption_sum, tz_name="Europe/Berlin") -> dict
   Computes PV KPI totals (kWh) and shares:
     - pv_feed_in_sum, pv_production_sum, pv_self_consumption_sum, pv_bat_sum
     - pv_self_consumption_share = pv_self / pv_production
     - pv_total_consumption_share = pv_self / (pv_self + grid_consumption_sum)
     - net_site_consumption_sum = ∑ net_consumption * hours
     - peak (kW) and peak_date ("DD.MM.YYYY") from "net_import" and "timestamp"

Usage
-----
These utilities are designed for reporting pipelines and notebooks
that analyze microgrid performance in hybrid systems with PV, batteries,
and grid imports/exports. They serve as a foundation for creating
dashboards, KPIs, and energy-mix summaries in localized formats.
"""

import os
from datetime import timedelta
from typing import Dict, Iterable, Tuple

import pandas as pd

from frequenz.data.microgrid.config import MicrogridConfig
from frequenz.lib.notebooks.reporting.utils.helpers import _fmt_de


def compute_energy_summary(df: pd.DataFrame, resolution: timedelta) -> pd.DataFrame:
    """Compute an energy-mix summary (PV vs CHP vs. grid).

    Aggregates energy over the given time resolution and reports
    totals in kWh, share in percent, and average power in kW. It expects
    ``grid_consumption`` (grid import) and optionally ``pv_self_consumption``
    (self-consumed PV) to exist in ``df``.

    Args:
      df: DataFrame containing energy data, with ``grid_consumption`` and
        optionally ``pv_self_consumption``.
      resolution: Row time resolution (e.g., ``"15min"`` or
        ``pd.Timedelta("0 days 00:15:00")``).

    Returns:
      pd.DataFrame: A summary table with columns:
        - ``Energy Source``: Source label ("PV" or "Grid" or "CHP").
        - ``Energy [kWh]``: Total energy per source.
        - ``Energy %``: Percentage share of total energy.
        - ``Power [kW]``: Average power over the interval.
    """
    resolution = pd.to_timedelta(resolution)
    hours = resolution.total_seconds() / 3600.0
    pv_kwh = 0.0
    grid_kwh = 0.0

    if "net_import" in df.columns:
        grid_kwh = df["net_import"].sum() * hours

    if "pv_self" in df.columns:
        pv_kwh = df["pv_self"].sum() * hours

    # Build rows (PV first if present)
    rows = []
    if pv_kwh > 0:
        rows.append(("PV", pv_kwh))
    if grid_kwh > 0 or not rows:
        rows.append(("Grid", grid_kwh))

    total_kwh = sum(v for _, v in rows)
    denom = total_kwh if total_kwh != 0 else 1.0  # zero-total guard

    data = {
        "Energy Source": [name for name, _ in rows],
        "Energy [kWh]": [round(val, 2) for _, val in rows],
        "Energy %": [round(val / denom * 100, 2) for _, val in rows],
        "Power [kW]": [round((val / hours) if hours else 0.0, 2) for _, val in rows],
    }
    return pd.DataFrame(data)


# pylint: disable=too-many-arguments, too-many-positional-arguments
def display_pv_energy(
    energy_report_df: pd.DataFrame,
    resolution: timedelta,
    label_contains: str = "PV #",
    include_total: bool = True,
    print_empty_message: bool = True,
    fmt_to_de: bool = False,
) -> dict[str, float]:
    """
    Summarize per-component PV energy from ``energy_report_df`` by column name pattern.

    Searches for columns whose names contain ``label_contains``
    (default: ``"PV #"``). For each matching column, it computes energy in kWh
    using the time-step ``resolution``. Values are printed with localized
    number formatting (comma as decimal separator). Optionally includes a
    total row labeled ``"PV Total"``.

    Args:
        energy_report_df: Table with PV component columns (power values per row).
        resolution: Time step per row (e.g., 15 minutes). Used to convert power sums to energy.
        label_contains: Substring used to identify PV component columns.
        include_total: If True, also prints/returns the total across all PV components.
        print_empty_message: If True, prints a friendly message when no PV columns are found.
        fmt_to_de: If True, use German number formatting (comma as decimal separator).

    Returns:
        Mapping of component label to energy in kWh. Includes ``"PV Total"`` when
        ``include_total`` is True.
    """
    # Find PV columns by name pattern
    pv_columns = [c for c in energy_report_df.columns if label_contains in c]

    if not pv_columns:
        if print_empty_message:
            print("No PV components found.")
        return {}

    step_hours = (
        (resolution.total_seconds() / 3600.0)
        if hasattr(resolution, "total_seconds")
        else (resolution.seconds / 3600.0)
    )

    results = {}
    # Convention: PV power is negative; multiply by -1 to report positive energy (kWh)
    for pv in pv_columns:
        pv_sum_kwh = round(energy_report_df[pv].sum() * step_hours * -1, 2)
        results[pv] = pv_sum_kwh
        val_str = _fmt_de(pv_sum_kwh) if fmt_to_de else f"{pv_sum_kwh:.2f}"
        print(f"{pv:<12}:   {val_str} kWh")

    if include_total:
        total_kwh = round(sum(results.values()), 2)
        results["PV Total"] = total_kwh
        val_str = _fmt_de(total_kwh) if fmt_to_de else f"{total_kwh:.2f}"
        print(f"{'PV Total':<12}:   {val_str} kWh")

    return results


def build_pv_analysis_df(
    energy_report_df: pd.DataFrame, pv_filter: Iterable[str]
) -> pd.DataFrame:
    """
    Build a normalized PV analysis table from ``energy_report_df``.

    Detects PV component columns (those starting with ``"PV #"``), filters them
    according to ``pv_filter`` (case-insensitive ``"all"`` selects all), and
    unpivots the selected columns into a long table with columns
    ``timestamp``, optional ``grid``, ``PV``, and
    ``pv_feedin``. If a grid column is present, its value is divided
    equally among the selected PV components. PV feed-in values are negated so
    production is positive, and the ``"PV "`` prefix is stripped from labels
    (e.g., ``"PV #1"`` → ``"#1"``).

    Args:
      energy_report_df: Source table containing PV component columns that start with
        ``"PV #"`` and, optionally, ``grid`` and
        ``timestamp``.
      pv_filter: Iterable of PV identifiers matching the suffix after ``"PV #"``
        (e.g., ``"1"``, ``"2"``). If any value equals ``"all"`` (case-insensitive),
        all PV columns are selected.

    Returns:
      A long-form DataFrame with columns:
        - ``timestamp``
        - (optional) ``grid`` (normalized per PV)
        - ``PV`` (component label without the ``"PV "`` prefix)
        - ``COLUMN_PV_FEEDIN`` (positive production)

      Returns an empty DataFrame if no PV columns are present or none match the filter.

    Examples:
      >>> # Assuming columns: 'timestamp', 'grid',
      ... # 'PV #1', 'PV #2' in energy_report_df
      >>> out = build_pv_analysis_df(energy_report_df, pv_filter=["1"])
      >>> set(out.columns) >= {"timestamp", "PV", "pv_feedin"}
      True
      >>> out_all = build_pv_analysis_df(energy_report_df, pv_filter=["all"])
    """
    # Find all PV columns in dataframe
    all_pv_cols = [c for c in energy_report_df.columns if c.startswith("PV #")]
    if not all_pv_cols:
        return pd.DataFrame()

    # Determine which PV columns to use
    if any(str(x).lower() == "all" for x in pv_filter):
        pv_columns = all_pv_cols
    else:
        requested = [f"PV {pv}" for pv in pv_filter]
        pv_columns = [c for c in requested if c in energy_report_df.columns]

    if not pv_columns:
        return pd.DataFrame()

    id_vars = ["timestamp"]
    if "grid" in energy_report_df.columns:
        id_vars.append("grid")

    df = energy_report_df[id_vars + pv_columns].copy()
    df = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=pv_columns,
        var_name="PV",
        value_name="pv_feedin",
    )

    # Adjust grid connection if present
    if "grid" in id_vars:
        df["grid"] /= len(pv_columns)

    # Common post-processing
    df["pv_feedin"] *= -1
    df["PV"] = df["PV"].str[3:]  # strip "PV "

    return df


def build_component_analysis(
    energy_report_df: pd.DataFrame,
    selection_filter: Iterable[str],
    component_label: str,
    value_col_name: str,
) -> pd.DataFrame:
    """
    Create a tidy analysis DataFrame for a single component type.

    Args:
        energy_report_df:
            DataFrame containing columns named like
            `"<component_label> #1"`, `"<component_label> #2"`, etc.
            Example: `"Battery #1"`, `"CHP #2"`, `"EV #3"`.
        selection_filter:
            - If it contains `"All"` (case-insensitive), all
              `"<component_label> #"` columns are selected.
            - Otherwise, should contain component numbers as strings
              starting with `"#"`, e.g., `["#1", "#3"]`.
        component_label:
            The label prefix used in the column names and the melted
            identifier column (e.g., `"Battery"`, `"CHP"`, `"EV"`).
        value_col_name:
            The output value column name for the melted values
            (e.g., `battery`, `chp`,
            `ev`).

    Returns:
        pd.DataFrame:
            Long-form DataFrame with columns:
            `timestamp`, `component_label`, `value_col_name`.
    """
    prefix = f"{component_label} #"

    # Select columns
    if any(str(x).lower() == "all" for x in selection_filter):
        comp_columns = [
            col for col in energy_report_df.columns if col.startswith(prefix)
        ]
    else:
        comp_columns = [
            f"{component_label} {x}"
            for x in selection_filter
            if f"{component_label} {x}" in energy_report_df.columns
        ]

    if not comp_columns:
        return pd.DataFrame(columns=["timestamp", component_label, value_col_name])

    id_vars = ["timestamp"]
    analyse_df = energy_report_df[id_vars + comp_columns].copy()

    # Melt to long form
    analyse_df = pd.melt(
        analyse_df,
        id_vars=id_vars,
        value_vars=comp_columns,
        var_name=component_label,
        value_name=value_col_name,
    )

    # Keep only the number after "<component_label> "
    analyse_df[component_label] = analyse_df[component_label].str.replace(
        f"{component_label} ", "", regex=False
    )

    return analyse_df


def build_overview_df(
    energy_report_df: pd.DataFrame, component_types: Iterable[str]
) -> pd.DataFrame:
    """Return a subset of ``energy_report_df`` with relevant columns.

    The selection includes base columns and optional ones depending
    on ``component_types``.

    Args:
        energy_report_df: Source DataFrame with energy data.
        component_types: Iterable of component types to include
            (e.g., {"pv", "battery"}).

    Returns:
        A subset of ``energy_report_df`` containing only the selected
        base and optional columns.
    """
    base_cols = ["timestamp", "net_import", "net_consumption"]

    optional_cols = {
        "pv": ["pv_prod", "pv_feedin"],
        "battery": ["battery_throughput"],
    }

    # Collect columns in order
    cols = base_cols[:]
    for comp, comp_cols in optional_cols.items():
        if comp in component_types:
            cols.extend(comp_cols)

    # Safe selection: avoid KeyError if a column is missing
    cols = list(pd.Index(cols).intersection(energy_report_df.columns, sort=False))

    return energy_report_df[cols]


def load_microgrid_configs(
    directory: str = "toml_directory/",
) -> Tuple[Dict[str, MicrogridConfig], list[int]]:
    """Load all .toml microgrid configuration files from a directory.

    Args:
        directory: Path to the directory containing .toml files.

    Returns:
        Sorted list of available microgrid IDs.

    Raises:
        FileNotFoundError: If no .toml files are found in the directory.
    """
    toml_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".toml")
    ]

    if not toml_files:
        raise FileNotFoundError(f"No .toml files found in {directory}.")

    configs: Dict[str, "MicrogridConfig"] = {}
    for toml_file in toml_files:
        configs.update(MicrogridConfig.load_configs(toml_file))

    available_microgrids = sorted(int(x) for x in configs)
    return configs, available_microgrids


def aggregate_pv_metrics(  # pylint: disable=too-many-locals
    energy_report_df: pd.DataFrame,
    resolution: timedelta,
    grid_consumption_sum: float,
    *,
    tz_name: str = "Europe/Berlin",
) -> dict[str, float | None | str]:
    """Compute photovoltaic (PV) summary metrics.

    Aggregates PV-related energy (kWh), shares, site consumption, and peak grid
    import from the given DataFrame at the specified time resolution.

    Args:
        energy_report_df: Input data. Expected columns (some optional):
            - 'pv_feedin'       : PV feed-in to the grid
            - 'pv_prod'         : PV production
            - 'pv_self'         : PV self-consumption
            - 'pv_in_bat'       : PV energy into the battery
            - 'net_consumption' : Site consumption supplied by the grid (optional)
            - 'net_import'      : Grid import power (used for peak)
            - 'timestamp'       : Timestamp for peak-date labeling (optional)
            Missing series are treated as zeros.
        resolution: Duration represented by each row. Used to convert power to energy.
        grid_consumption_sum: Total grid consumption (kWh) over the same period.
        tz_name: Target timezone name for timestamp conversion (default: "Europe/Berlin").

    Returns:
        Dictionary containing the following metrics:
            - ``pv_feed_in_sum``: PV feed-in energy (kWh).
            - ``pv_production_sum``: Total PV production (kWh).
            - ``pv_self_consumption_sum``: Self-consumed PV energy (kWh).
            - ``pv_bat_sum``: PV energy into the battery (kWh).
            - ``pv_self_consumption_share``: Self-consumption / production (0-1).
            - ``pv_total_consumption_share``: Self-consumption / total site consumption (0-1).
            - ``net_site_consumption_sum``: Site consumption supplied by the grid (kWh).
            - ``peak``: Peak grid import (kW).
            - ``peak_date``: Date of peak import in ``DD.MM.YYYY`` or ``None``.
    """
    hours_factor = resolution.total_seconds() / 3600.0

    # Always get columns safely (Series of zeros if missing)
    zeros = pd.Series(0, index=energy_report_df.index)
    pv_feed_in = energy_report_df.get("pv_feedin", zeros)
    pv_production = energy_report_df.get("pv_prod", zeros)
    pv_self = energy_report_df.get("pv_self", zeros)
    pv_bat = energy_report_df.get("pv_in_bat", zeros)

    # Energy sums in kWh
    pv_feed_in_sum = (pv_feed_in * hours_factor).sum()
    pv_production_sum = (pv_production * hours_factor).sum()
    pv_self_consumption_sum = (pv_self * hours_factor).sum()
    pv_bat_sum = (pv_bat * hours_factor).sum()

    # Shares
    pv_self_consumption_share = (
        pv_self_consumption_sum / pv_production_sum if pv_production_sum > 0 else 0
    )
    total_consumed = pv_self_consumption_sum + grid_consumption_sum
    pv_total_consumption_share = (
        pv_self_consumption_sum / total_consumed if total_consumed > 0 else 0
    )

    # Always compute site consumption + peak
    net_site_consumption_sum = float(
        (energy_report_df.get("net_consumption", zeros).sum()) * hours_factor
    )
    peak_series = energy_report_df.get("net_import", zeros)
    peak = float(peak_series.max()) if not peak_series.empty else 0.0

    peak_date = None
    if "net_import" in energy_report_df.columns and not peak_series.empty:
        peak_idx = peak_series.idxmax()
        if "timestamp" in energy_report_df.columns:
            ts_raw = energy_report_df.loc[peak_idx, "timestamp"]
            ts = pd.to_datetime(str(ts_raw), utc=True, errors="coerce")
            peak_date = (
                ts.tz_convert(tz_name).strftime("%d.%m.%Y") if not pd.isna(ts) else None
            )

    return {
        "pv_feed_in_sum": pv_feed_in_sum,
        "pv_production_sum": pv_production_sum,
        "pv_self_consumption_sum": pv_self_consumption_sum,
        "pv_bat_sum": pv_bat_sum,
        "pv_self_consumption_share": pv_self_consumption_share,
        "pv_total_consumption_share": pv_total_consumption_share,
        "net_site_consumption_sum": net_site_consumption_sum,
        "peak": peak,
        "peak_date": peak_date,
    }
