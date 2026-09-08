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
from typing import Iterable, Literal, Union, cast

import pandas as pd

from frequenz.data.microgrid import MicrogridConfig
from .column_mapper import ColumnMapper


def _extract_component_id(component_label: str, column_name: str) -> str | None:
    """Extract a component ID from either legacy or display-name labels."""
    prefix_with_hash = f"{component_label} #"
    if not column_name.startswith(prefix_with_hash):
        return None

    remainder = column_name[len(prefix_with_hash) :]
    component_id = remainder.split(" ", maxsplit=1)[0]
    return component_id if component_id.isdigit() else None


def _format_component_value(component_label: str, column_name: str) -> str:
    """Convert a component column name into the final display value."""
    component_id = _extract_component_id(component_label, column_name)
    if component_id is None:
        return column_name

    remainder = column_name.split(f"#{component_id}", maxsplit=1)[1].strip()
    if not remainder:
        return f"#{component_id}"

    if remainder.startswith("-"):
        return f"{component_label} #{component_id} {remainder}"

    return remainder


# pylint: disable=too-many-arguments, too-many-positional-arguments
def _select_component_columns(
    energy_report_df: pd.DataFrame,
    selection_filter: Iterable[str],
    component_label: str,
    allowed_component_ids: set[str] | None = None,
) -> list[str]:
    """Resolve component columns from user selection and optional ID constraints."""
    prefix = f"{component_label} #"
    comp_columns: list[str] = []
    selected_components = list(selection_filter)

    if any(str(x).lower() == "all" for x in selected_components):
        comp_columns = [
            col for col in energy_report_df.columns if col.startswith(prefix)
        ]
    else:
        seen_columns: set[str] = set()
        for selected_component in selected_components:
            normalized_selection = str(selected_component).strip()
            selected_prefix = (
                f"{component_label} {normalized_selection}"
                if not normalized_selection.startswith(f"{component_label} #")
                else normalized_selection
            )
            for col in energy_report_df.columns:
                if col in seen_columns or not col.startswith(prefix):
                    continue

                if col == selected_prefix or col.startswith(f"{selected_prefix} "):
                    comp_columns.append(col)
                    seen_columns.add(col)
                    continue

                if (
                    _format_component_value(component_label, col)
                    == normalized_selection
                ):
                    comp_columns.append(col)
                    seen_columns.add(col)

    if allowed_component_ids is None:
        return comp_columns

    return [
        col
        for col in comp_columns
        if _extract_component_id(component_label, col) in allowed_component_ids
    ]


def build_component_analysis(
    energy_report_df: pd.DataFrame,
    selection_filter: Iterable[str],
    component_label: str,
    value_col_name: str,
    allowed_component_ids: set[str] | None = None,
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
        allowed_component_ids:
            Optional set of component IDs to include in the analysis.

    Returns:
        pd.DataFrame:
            A long-format DataFrame with columns:
                - "timestamp"
                - <component_label>: containing only the numeric component ID
                - <value_col_name>: containing the component's values

            If no matching columns are found, returns an empty DataFrame with
            the appropriate columns.
    """
    comp_columns = _select_component_columns(
        energy_report_df,
        selection_filter,
        component_label,
        allowed_component_ids,
    )

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

    analyse_df[component_label] = analyse_df[component_label].map(
        lambda column_name: _format_component_value(component_label, column_name)
    )

    return analyse_df


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
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
    mcfg: MicrogridConfig | None = None,
    component_id_source: Literal["meter", "inverter"] | None = None,
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
            Component type key (e.g. ``"pv"``, ``"battery"``, ``"chp"``, ``"wind"``)
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
        mcfg:
            Optional microgrid config used to resolve component IDs.
        component_id_source:
            Optional ID source selector. Use ``"meter"`` or ``"inverter"``.
            Leave unset to keep the previous unrestricted behavior.

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

    allowed_component_ids: set[str] | None = None
    if mcfg is not None and component_id_source is not None:
        try:
            component_ids = mcfg.component_type_ids(
                component_key, component_category=component_id_source
            )
        except ValueError:
            component_ids = []
        allowed_component_ids = {str(cid) for cid in component_ids or []}

    # Build Analysis
    analyse_df = build_component_analysis(
        energy_report_df,
        component_filter,
        component_label=component_label,
        value_col_name=value_col_name,
        allowed_component_ids=allowed_component_ids,
    )

    analyse_df[value_col_name] = pd.to_numeric(
        analyse_df[value_col_name],
        errors="coerce",
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
    energy_report_df: pd.DataFrame,
    component_types: Iterable[str],
) -> pd.DataFrame:
    """Build an overview table from a canonical energy report DataFrame.

    The output always starts with the site-level columns ``timestamp``,
    ``grid_consumption``, ``mid_consumption``, and ``grid_feed_in`` when they
    are available in ``energy_report_df``. It then appends component-level
    aggregate columns for the requested component types:
    ``pv_asset_production``, ``chp_asset_production``,
    ``wind_asset_production``, ``battery_power_flow``, and
    ``battery_soc_pct`` when present.

    If ``battery`` is present, the output is extended with battery plotting
    helpers derived from the selected columns: ``peak_before_optimization``,
    ``peak_after_optimization``, ``battery_charge``, and
    ``battery_discharge``.

    Args:
        energy_report_df: Canonical energy report DataFrame.
        component_types: Component type identifiers to include, e.g.
            ``{"pv", "battery", "wind"}``.

    Returns:
        A copy of the selected overview columns plus battery helper columns when
        ``battery`` is present.
    """
    base_cols = [
        "timestamp",
        "grid_consumption",
        "mid_consumption",
        "grid_feed_in",
        "day_ahead_price",
    ]

    optional_cols = {
        "pv": ["pv_asset_production"],
        "chp": ["chp_asset_production"],
        # "ev": ["ev_charging_load"],
        "wind": ["wind_asset_production"],
        "battery": ["battery_power_flow", "battery_soc_pct"],
    }

    # Collect columns in order
    cols = base_cols[:]
    for comp, comp_cols in optional_cols.items():
        if comp in component_types:
            cols.extend(comp_cols)

    # Safe selection: avoid KeyError if a column is missing
    cols = list(pd.Index(cols).intersection(energy_report_df.columns, sort=False))

    overview_df = energy_report_df[cols].copy()
    overview_df.columns = overview_df.columns.str.replace(
        "_asset_production", "", regex=False
    )

    if "battery" in component_types:
        overview_df["peak_before_optimization"] = overview_df["mid_consumption"].max()
        overview_df["peak_after_optimization"] = overview_df["grid_consumption"].max()
        overview_df["battery_charge"] = overview_df["battery_power_flow"].clip(lower=0)
        overview_df["battery_discharge"] = overview_df["battery_power_flow"].clip(
            upper=0
        )

    return overview_df


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
            - ``grid_consumption`` — Grid power, clipped to positive import values
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
    data["grid_consumption"] = data["grid_consumption"].clip(lower=0)

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


# pylint: disable=too-many-locals, too-many-statements
def aggregate_metrics(
    energy_report_df: pd.DataFrame,
    resolution: timedelta,
    *,
    tz_name: str = "Europe/Berlin",
    price_column: str | None = None,
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
            - ``battery_power_flow`` — Battery power using PSC convention:
              positive values charge the battery, negative values discharge it
            - ``production_to_battery`` — Production power stored in the battery
            - ``grid_to_battery`` — Grid power stored in the battery
            - ``battery_to_grid`` — Battery discharge exported to the grid
            - ``battery_to_consumption`` — Battery discharge used locally
            - ``grid_feed_in`` — Exported power to the grid
            - ``grid_consumption`` — Imported power from the grid
            - ``mid_consumption`` — Total site consumption power
        resolution:
            Sampling interval between measurements (e.g. ``timedelta(minutes=15)``).
            Used to convert power values into energy (kWh).
        tz_name:
            Timezone used when reporting the date of the peak grid consumption.
            Defaults to ``"Europe/Berlin"``.
        price_column:
            Name of the column containing the day-ahead price information.
            Defaults to ``"day_ahead_price"``.

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
            Alias for ``production_to_battery_sum`` for backward compatibility
            - ``production_to_battery_sum``
            Production energy stored in the battery (kWh)
            - ``grid_to_battery_sum``
            Grid energy stored in the battery (kWh)
            - ``battery_to_grid_sum``
            Battery discharge exported to the grid (kWh)
            - ``battery_to_consumption_sum``
            Battery discharge used by local consumption (kWh)
            - ``grid_feed_in_sum``
            Total exported energy (kWh)
            - ``grid_consumption_sum``
            Total imported energy (kWh)
            - ``mid_consumption_sum``
            Total site consumption (kWh)
            - ``prod_self_consumption_share``
            Fraction of site consumption covered by self-production (0-1)
            - ``production_self_share``
            Fraction of total production that is self-consumed (0-1)
            - ``production_self_usage``
            Fraction of site consumption covered by self-production (0-1)
            - ``peak``
            Maximum grid import power (kW)
            - ``peak_date``
            Localized date (``DD.MM.YYYY``) of grid-import peak, or ``None``
            - ``grid_import_cost_sum``
            Total import cost based on spot/day-ahead prices (EUR)
            - ``grid_feed_in_revenue_sum``
            Total export revenue based on spot/day-ahead prices (EUR)

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
        "grid_feed_in": "grid_feed_in_sum",
        "grid_consumption": "grid_consumption_sum",
        "mid_consumption": "mid_consumption_sum",
        "production_to_battery": "production_to_battery_sum",
        "grid_to_battery": "grid_to_battery_sum",
        "battery_to_grid": "battery_to_grid_sum",
        "battery_to_consumption": "battery_to_consumption_sum",
    }

    results = {}
    zeros = pd.Series(0, index=energy_report_df.index)

    # Energy Sums Calculation (kWh)
    for col_name, result_key in metric_columns.items():
        # Safely get column (or Series of zeros if missing)
        series = energy_report_df.get(col_name, zeros)

        # Clip grid consumption to non-negative values to avoid negative grid feed-in values
        if col_name == "grid_consumption":
            series = series.clip(lower=0)

        # Calculate energy sum (Power * hours_factor)
        results[result_key] = (series * hours_factor).sum()

    # Calculate derived total production sum
    total_production_sum = (
        results.get("pv_production_sum", 0)
        + results.get("chp_production_sum", 0)
        + results.get("wind_production_sum", 0)
    )
    results["total_production_sum"] = total_production_sum
    results["prod_bat_sum"] = results["production_to_battery_sum"]

    # Note: Use the new consistent key for total consumption
    total_consumption_sum = results.get("mid_consumption_sum", 0)
    prod_self_consumption_sum = results.get("prod_self_consumption_sum", 0)
    results["prod_self_consumption_share"] = (
        prod_self_consumption_sum / total_consumption_sum
        if total_consumption_sum > 0
        else 0
    )

    results["prod_self_production_share"] = (
        prod_self_consumption_sum / total_production_sum
        if total_production_sum > 0
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
    if price_column is None:
        results["grid_import_cost_sum"] = 0.0
        results["grid_feed_in_revenue_sum"] = 0.0
        return results

    prices_eur_mwh = energy_report_df[price_column].fillna(0.0).astype(float)
    # Assuming the grid consumption and grid_feed_in is in kW and not kWh
    grid_import_energy_kwh = (
        energy_report_df.get("grid_consumption", zeros).clip(lower=0) * hours_factor
    )
    grid_feed_in_energy_kwh = (
        energy_report_df.get("grid_feed_in", zeros).clip(lower=0) * hours_factor
    )

    # Day-ahead and spot prices are expected in EUR/MWh. Convert kWh to MWh first.
    results["grid_import_cost_sum"] = float(
        (grid_import_energy_kwh * (prices_eur_mwh / 1000.0)).sum()
    )
    results["grid_feed_in_revenue_sum"] = float(
        (grid_feed_in_energy_kwh * (prices_eur_mwh / 1000.0)).sum()
    )

    return results
