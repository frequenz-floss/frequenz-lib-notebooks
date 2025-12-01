# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Utilities for analyzing and summarizing microgrid energy-flow data.

This module contains helper functions for transforming, aggregating, and
summarizing power and energy data used throughout the reporting pipeline.
It supports component-level analysis, overview table construction, energy
summaries, and site-level metric aggregation.

Functions
---------
- ``build_component_analysis()``
    Produce a tidy (long-format) DataFrame for a specific component type
    (e.g., Battery, PV, CHP), based on selected component IDs.

- ``build_overview_df()``
    Select and return only the relevant reporting columns for overview plots,
    depending on available component types.

- ``compute_energy_summary()``
    Compute relative distribution among production and grid consumption
    sources.

- ``aggregate_metrics()``
    Compute high-level site metrics including production totals, self-
    consumption energy and share, battery-related flows, grid import and
    feed-in, and peak grid consumption with localized date.

Usage
-----
These functions are typically applied to DataFrames produced by the
normalized Energy Report pipeline. Input columns are assumed to represent
instantaneous power measurements (in kW) sampled at a known fixed resolution.
Energy values (kWh) are derived by multiplying power samples by the sampling
interval.

Typical workflow:
1. Build an energy report DataFrame upstream (e.g., via `create_energy_report_df`).
2. Use ``build_overview_df`` to extract relevant columns for dashboards.
3. Use ``build_component_analysis`` to analyze per-component contributions.
4. Use ``compute_energy_summary`` to generate energy-mix tables.
5. Use ``aggregate_metrics`` to calculate site-wide KPIs such as production
   totals, self-consumption share, and grid import peaks.

All missing or unavailable columns are treated safely (as zero-valued Series),
ensuring resilient operation even with partially populated datasets.
"""


from datetime import datetime, timedelta
from typing import Iterable, Union, cast

import pandas as pd

from .column_mapper import ColumnMapper


def build_component_analysis(
    energy_report_df: pd.DataFrame,
    selection_filter: Iterable[str],
    component_label: str,
    value_col_name: str,
) -> pd.DataFrame:
    """Build a long-format analysis table for a single component type.

    Selects component columns such as "<component_label> #1", "<component_label> #2",
    etc., according to the provided filter, and reshapes them into a tidy,
    long-format DataFrame suitable for plotting or comparative analysis.

    Args:
        energy_report_df:
            DataFrame containing timestamped component data with columns named in the
            form "<component_label> #<id>" (e.g., "Battery #1", "CHP #2", "EV #3").
        selection_filter:
            Iterable defining which components to include:
            - If any entry equals "All" (case-insensitive), all matching component
            columns are selected.
            - Otherwise, entries should be component identifiers such as ["#1", "#3"].
        component_label:
            The base label used in the component column names and in the resulting
            identifier column (e.g., "Battery", "CHP", "EV").
        value_col_name:
            Name of the output column containing the selected component data
            (e.g., "battery", "chp", "ev").

    Returns:
        pd.DataFrame:
            A long-format DataFrame with columns:
                - "timestamp"
                - <component_label>: containing only the numeric component ID
                - <value_col_name>: containing the component's values

            If no matching columns are found, returns an empty DataFrame with
            the appropriate columns.
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


# pylint: disable=too-many-arguments, too-many-positional-arguments
def assemble_component_analysis(
    component_filter: list[str],
    component_key: str,
    component_types: list[str],
    energy_report_df: pd.DataFrame,
    timestep_hours: float,
    mapper: ColumnMapper,
    component_label: str,
    value_col_name: str,
    invert_sign: bool = False,
    trunc_values: bool = False,
) -> tuple[pd.DataFrame, float, str]:
    """Assemble a component-level analysis table and compute its energy total.

    This function retrieves one or more component columns from the
    Energy Report DataFrame (e.g., individual PV strings, batteries, CHP
    units), converts them into long-form using ``build_component_analysis()``,
    applies display-name mapping, scales values by the timestep duration,
    optionally inverts the sign, and returns both the transformed DataFrame
    and the aggregated energy.

    Args:
        component_filter:
            List of component selectors. Can contain component numbers
            (e.g. ``["#1", "#3"]``) or ``"All"``/``"Alle"`` to include all.
        component_key:
            Component type key (e.g. ``"pv"``, ``"battery"``, ``"chp"``)
            used to check availability inside ``component_types``.
        component_types:
            List of component types present in the Energy Report.
        energy_report_df:
            Source DataFrame containing timestamped component data.
            Must include a ``"timestamp"`` column.
        timestep_hours:
            Sampling interval expressed in hours (e.g. ``0.25`` for 15 min).
            Used to convert instantaneous values (kW) to energy (kWh).
        mapper:
            ColumnMapper used to convert column names into display labels.
        component_label:
            Human-readable label to inject into the melted output
            (e.g., ``"Battery"``, ``"PV"``, ``"CHP"``).
        value_col_name:
            Name of the value column in the melted long-format DataFrame.
        invert_sign:
            Whether to multiply results by ``-1`` after scaling. Useful when
            converting export flows to positive energy values.
        trunc_values:
            If ``True``, truncate values to zero (i.e., set negative values to zero)
            after scaling (dataframe values are kept unchanged).

    Returns:
        - The long-form analysis DataFrame with energy values in kWh.
        - The total summed energy (rounded to two decimals).
        - A textual representation of the component filter used.

    Notes:
        - If the component type is not present or the DataFrame lacks a
        ``"timestamp"`` column, an empty result is returned.
    """
    analyse_df = pd.DataFrame()
    component_sum = 0

    # Normalize filters
    if "Alle" in component_filter or "All" in component_filter:
        component_filter = ["All"]
    filter_text = ",".join(component_filter)

    # Check failure conditions first to avoid deep nesting
    if (
        component_key not in component_types
        or "timestamp" not in energy_report_df.columns
    ):
        return pd.DataFrame(), 0, filter_text

    # Build Analysis
    analyse_df = build_component_analysis(
        energy_report_df,
        component_filter,
        component_label=component_label,
        value_col_name=value_col_name,
    )

    # Calculate timestep scaling according to resolution
    factor = timestep_hours * (-1 if invert_sign else 1)

    # Apply timestep scaling
    analyse_df[value_col_name] *= factor

    # Summation - Optionally drop small positive values seen in production data
    # (e.g., due to measurement noise) - we have already inverted the sign for production
    # so now negative values need to be truncated.
    sum_series = analyse_df[value_col_name].copy()
    if trunc_values:
        sum_series = sum_series[sum_series >= 0]
    component_sum = round(sum_series.sum(), 3)

    # Round for readability
    analyse_df[value_col_name] = analyse_df[value_col_name].round(3)

    # Rename for output clarity
    # new_col_name = f"{component_label}-Energie [kWh]"
    # analyse_df.rename(columns={value_col_name: new_col_name}, inplace=True)

    analyse_df = mapper.to_display(analyse_df)

    return analyse_df, component_sum, filter_text


def build_overview_df(
    energy_report_df: pd.DataFrame, component_types: Iterable[str]
) -> pd.DataFrame:
    """Return a compact overview subset of the energy report DataFrame.

    This function extracts a core set of site-level energy columns together with
    optional production-related columns, depending on which component types are
    present (e.g., PV, battery, wind, CHP). Missing columns are ignored safely.

    Args:
        energy_report_df:
            The full energy report DataFrame containing timestamped power data and
            optionally component-specific production/throughput columns.
        component_types:
            Iterable of component type identifiers (e.g., {"pv", "battery", "wind"}).
            Only columns corresponding to these component types are included.

    Returns:
        pd.DataFrame:
            A subset of `energy_report_df` that contains:
                - Base columns: "timestamp", "grid_consumption",
                "mid_consumption", "grid_feed_in"
                - Optional component-specific columns such as
                "pv_asset_production", "battery_power_flow",
                "chp_asset_production", "wind_asset_production"

            Columns that do not exist in the input DataFrame are silently skipped.
    """
    base_cols = [
        "timestamp",
        "grid_consumption",
        "mid_consumption",
        "grid_feed_in",
    ]

    optional_cols = {
        "pv": ["pv_asset_production"],
        "chp": ["chp_asset_production"],
        # "ev": ["ev_charging_load"],
        "wind": ["wind_asset_production"],
        "battery": ["battery_power_flow"],
    }

    # Collect columns in order
    cols = base_cols[:]
    for comp, comp_cols in optional_cols.items():
        if comp in component_types:
            cols.extend(comp_cols)

    # Safe selection: avoid KeyError if a column is missing
    cols = list(pd.Index(cols).intersection(energy_report_df.columns, sort=False))

    return energy_report_df[cols]


# pylint: disable=too-many-locals
def compute_energy_summary(
    df: pd.DataFrame,
    resolution: timedelta,
    include_rollups: bool = False,
    drop_zeros: bool = True,
) -> pd.DataFrame:
    """
    Compute energy totals, average power, and percentage shares for key energy sources.

    This function aggregates instantaneous power measurements (kW) over a fixed
    sampling interval to produce energy statistics (kWh) for major sources such as
    PV, wind, CHP, and grid consumption. It supports optional roll-ups of total
    on-site production and configurable filtering of near-zero results.

    Args:
        df:
            Input DataFrame containing instantaneous power columns in kW. Only the
            following canonical columns are considered when present and numeric:
            - ``pv_asset_production`` — Photovoltaic production
            - ``wind_asset_production`` — Wind turbine production
            - ``chp_asset_production`` — CHP unit production
            - ``grid_consumption`` — Grid import power
        resolution:
            Sampling interval between observations (e.g. ``timedelta(minutes=15)``),
            used to convert power values (kW) into energy (kWh).
        include_rollups:
            If ``True``, adds a combined row “Production (PV+Wind+CHP)” summarizing
            all on-site generation. Default: ``False``.
        drop_zeros:
            If ``True``, removes rows where the computed energy is effectively zero
            (|Energy [kWh]| < 1e-9). Default: ``True``.

    Returns:
        pd.DataFrame:
            A summary table with one row per included energy source and columns:
            - ``Energy Source`` — Human-readable label (e.g., “PV”, “Wind”)
            - ``Energy [kWh]`` — Total energy over the reporting period
            - ``Power [kW]`` — Sum of instantaneous samples
            - ``Mean [kW]`` — Average power over the reporting window
            - ``Energy %`` — Percentage contribution relative to total energy

    Raises:
        ValueError:
            If ``resolution`` is non-positive.
    """
    # Convert sampling resolution to hours; validate positivity.
    dt_h = float(pd.to_timedelta(resolution).total_seconds()) / 3600.0
    if dt_h <= 0:
        raise ValueError("resolution must be positive")

    # Determine total duration represented by the dataframe.
    n_steps = len(df)
    total_time_h = n_steps * dt_h if n_steps else 0.0

    # Canonical column → display name mapping.
    mapping = {
        "pv_asset_production": "PV",
        "wind_asset_production": "Wind",
        "chp_asset_production": "CHP",
        "grid_consumption": "Grid Consumption",
    }

    # Restrict to known, numeric columns that actually exist in the input.
    known = list(mapping.keys())
    cols = [
        c for c in known if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        return pd.DataFrame(
            columns=[
                "Energy Source",
                "Energy [kWh]",
                "Power [kW]",
                "Mean [kW]",
                "Energy %",
            ]
        )

    # Replace NaNs with 0 for safe aggregation.
    data = df[cols].fillna(0.0)

    # Vectorized totals
    # Sum of instantaneous samples (kW) across all timesteps for each column
    sum_samples_kw = data.sum(axis=0).astype(float)

    # Energy over the period
    energy_kwh = sum_samples_kw * dt_h

    # Mean power over the reporting window; guard against empty input.
    mean_kw = (energy_kwh / total_time_h) if total_time_h > 0 else sum_samples_kw * 0.0

    # Assemble the per-source summary table.
    out = pd.DataFrame(
        {
            "Energy Source": [mapping[c] for c in cols],
            "Energy [kWh]": energy_kwh.reindex(cols).values,
            "Power [kW]": sum_samples_kw.reindex(cols).values,
            "Mean [kW]": mean_kw.reindex(cols).values,
        }
    )

    # Optional roll-up: on-site production (PV + Wind + CHP)
    if include_rollups:
        prod_cols = [
            c
            for c in (
                "pv_asset_production",
                "wind_asset_production",
                "chp_asset_production",
            )
            if c in cols
        ]
        if prod_cols:
            prod_series_kw = data[prod_cols].sum(axis=1)
            prod_sum_kw = float(prod_series_kw.sum())
            prod_kwh = prod_sum_kw * dt_h
            prod_mean_kw = (prod_kwh / total_time_h) if total_time_h > 0 else 0.0
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        [
                            {
                                "Energy Source": "Production (PV+Wind+CHP)",
                                "Energy [kWh]": prod_kwh,
                                "Power [kW]": prod_sum_kw,
                                "Mean [kW]": prod_mean_kw,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # Optionally remove near-zero energy rows to cut numerical noise.
    if drop_zeros:
        out = out[~out["Energy [kWh]"].abs().lt(1e-9)]

    # If everything was filtered out, return a typed empty frame.
    if out.empty:
        return pd.DataFrame(
            columns=[
                "Energy Source",
                "Energy [kWh]",
                "Power [kW]",
                "Mean [kW]",
                "Energy %",
            ]
        )

    # Compute percentage share from absolute energies to avoid sign cancellation.
    denom = out["Energy [kWh]"].abs().sum() or 1.0
    out["Energy %"] = (out["Energy [kWh]"].abs() / denom * 100.0).round(3)

    # Round numeric display columns for readability.
    out[["Energy [kWh]", "Power [kW]", "Mean [kW]"]] = out[
        ["Energy [kWh]", "Power [kW]", "Mean [kW]"]
    ].round(3)

    # Stable, human-friendly ordering for output rows.
    order_hint = {
        name: i
        for i, name in enumerate(
            ["PV", "Wind", "CHP", "Production (PV+Wind+CHP)", "Grid Consumption"]
        )
    }
    out = out.sort_values(
        by="Energy Source", key=lambda s: s.map(order_hint).fillna(999)
    ).reset_index(drop=True)
    return out


def aggregate_metrics(  # pylint: disable=too-many-locals
    energy_report_df: pd.DataFrame,
    resolution: timedelta,
    *,
    tz_name: str = "Europe/Berlin",
) -> dict[str, float | None | str]:
    """Aggregate key site-level energy and performance metrics from time-series data.

    This function converts instantaneous power measurements (kW) into energy values
    (kWh) using the given sampling resolution and computes aggregated indicators
    across all major energy sources: PV, CHP, Wind, Battery, Grid, and total site
    consumption. It also evaluates self-consumption ratios and determines the peak
    grid consumption including its calendar date.

    Args:
        energy_report_df:
            DataFrame containing time-series power data (kW). Missing columns are
            treated as zero. Expected canonical column names include:
            - ``pv_asset_production`` — PV generation power
            - ``chp_asset_production`` — CHP generation power
            - ``wind_asset_production`` — Wind generation power
            - ``production_self_use`` — Self-consumed on-site production
            - ``production_excess_in_bat`` — Excess production stored in the battery
            - ``grid_feed_in`` — Exported power to the grid
            - ``grid_consumption`` — Imported power from the grid
            - ``mid_consumption`` — Total site consumption power
        resolution:
            Sampling interval between measurements (e.g. ``timedelta(minutes=15)``).
            Used to convert power values into energy (kWh).
        tz_name:
            Timezone used when reporting the date of the peak grid consumption.
            Defaults to ``"Europe/Berlin"``.

    Returns:
        dict[str, float | None | str]:
            A dictionary of aggregated metrics including:
            - ``pv_production_sum`` / ``chp_production_sum`` / ``wind_production_sum``
            Total production per source (kWh)
            - ``total_production_sum``
            Combined production (kWh)
            - ``prod_self_consumption_sum``
            Total self-consumed energy (kWh)
            - ``prod_bat_sum``
            Energy stored in the battery (kWh)
            - ``grid_feed_in_sum``
            Total exported energy (kWh)
            - ``grid_consumption_sum``
            Total imported energy (kWh)
            - ``mid_consumption_sum``
            Total site consumption (kWh)
            - ``prod_self_consumption_share``
            Fraction of site consumption covered by self-production (0-1)
            - ``peak``
            Maximum grid import power (kW)
            - ``peak_date``
            Localized date (``DD.MM.YYYY``) of grid-import peak, or ``None``

    Raises:
        ValueError:
            If ``resolution`` is non-positive.

    Notes:
        - Missing columns are treated as zero-valued Series.
        - Peak date is determined from the index label of the maximum grid import.
        - Naive timestamps are assumed to be in UTC before timezone conversion.
    """
    hours_factor = float(pd.to_timedelta(resolution).total_seconds()) / 3600.0
    if hours_factor <= 0:
        raise ValueError("resolution must be positive")

    # Define all relevant columns and the corresponding key for the result dictionary
    metric_columns = {
        "pv_asset_production": "pv_production_sum",
        "chp_asset_production": "chp_production_sum",
        "wind_asset_production": "wind_production_sum",
        "production_self_use": "prod_self_consumption_sum",
        "production_excess_in_bat": "prod_bat_sum",
        "grid_feed_in": "grid_feed_in_sum",
        "grid_consumption": "grid_consumption_sum",
        "mid_consumption": "mid_consumption_sum",
    }

    results = {}
    zeros = pd.Series(0, index=energy_report_df.index)

    # Energy Sums Calculation (kWh)
    for col_name, result_key in metric_columns.items():
        # Safely get column (or Series of zeros if missing)
        series = energy_report_df.get(col_name, zeros)
        # Calculate energy sum (Power * hours_factor)
        results[result_key] = (series * hours_factor).sum()

    # Calculate derived total production sum
    total_production_sum = (
        results.get("pv_production_sum", 0)
        + results.get("chp_production_sum", 0)
        + results.get("wind_production_sum", 0)
    )
    results["total_production_sum"] = total_production_sum

    # Note: Use the new consistent key for total consumption
    total_consumption_sum = results.get("mid_consumption_sum", 0)
    prod_self_consumption_sum = results.get("prod_self_consumption_sum", 0)
    results["prod_self_consumption_share"] = (
        prod_self_consumption_sum / total_consumption_sum
        if total_consumption_sum > 0
        else 0
    )

    # Get the grid_consumption series for peak calculation (kW)
    grid_consumption_series = energy_report_df.get("grid_consumption", zeros)

    # Calculate peak power (kW)
    peak = (
        float(grid_consumption_series.max())
        if not grid_consumption_series.empty
        else 0.0
    )
    results["peak"] = peak

    peak_date = None
    if peak > 0 and not grid_consumption_series.empty:
        # idxmax returns the index label of the max; often a pd.Timestamp already
        peak_idx = grid_consumption_series.idxmax()

        if "timestamp" in energy_report_df.columns:
            raw_ts = energy_report_df.loc[peak_idx, "timestamp"]

            # Cast it to a type that pd.to_datetime accepts (str, float, or datetime)
            ts_input = cast(Union[str, float, datetime], raw_ts)

            # Ensure it is a datetime object
            ts = pd.to_datetime(ts_input)

            if pd.notna(ts):
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                peak_date = ts.tz_convert(tz_name).strftime("%d.%m.%Y")

    results["peak_date"] = peak_date

    return results
