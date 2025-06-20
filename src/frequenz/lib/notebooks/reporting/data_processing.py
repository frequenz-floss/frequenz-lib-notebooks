# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Data processing utilities for microgrid energy reporting.

This module provides a set of functions for processing, enriching, and analyzing
time-series energy data from microgrid systems. It focuses on preparing data for
PV (photovoltaic), battery, and grid energy flows, transforming it into a consistent
structure for visualization, reporting, and analysis.

Features
--------
- Enriches raw energy data with derived columns such as:
  - PV production, self-consumption, feed-in, and battery charging.
  - Net grid import and PV self-consumption share.
- Handles time zone localization and conversion to Europe/Berlin.
- Dynamically renames columns to more descriptive names, including
  mapping component IDs (e.g., "PV #1", "Batterie #2").
- Provides summary energy mix breakdowns (PV vs grid) in kWh, % share, and average kW.
- Prepares tailored DataFrames for PV and battery analysis, supporting flexible
  filtering by component.

Main Functions
--------------
- `transform_energy_dataframe(df, component_types, mcfg)`:
    Transforms a raw DataFrame with energy metrics into an enriched,
    user-friendly format, adding PV, battery, and grid metrics.

- `compute_power_df(main_df, resolution)`:
    Computes total energy drawn from PV and grid sources over the given resolution,
    returning a summary DataFrame with kWh, percentage, and average kW.

- `print_pv_sums(main_df, resolution)`:
    Prints total PV feed-in sums for each individual PV component
    in a localized numeric format.

- `create_pv_analysis_df(main_df, pv_filter, pvgrid_filter, pv_grid_filter_options)`:
    Generates a DataFrame for PV analysis based on selected PV components
    and whether to analyze PV alone, grid alone, or a grid/PV split.

- `create_battery_analysis_df(main_df, bat_filter)`:
    Creates a DataFrame for analyzing battery throughput, reshaping
    it to long format for multi-battery analysis.

Usage
-----
Typical usage involves:
1. Loading a raw DataFrame with time-indexed energy measurements.
2. Calling `transform_energy_dataframe` to process and enrich it.
3. Using the resulting DataFrames to generate summaries,
   for example with `compute_power_df`, `create_pv_analysis_df`, or
   `create_battery_analysis_df` for visualization.
"""

from typing import Any, Dict, Iterable, List, Tuple, Union

import pandas as pd

# Constants
TZ_NAME = "Europe/Berlin"
COLUMN_TIMESTAMP = "timestamp"
COLUMN_TIMESTAMP_NAMED = "Zeitpunkt"
COLUMN_GRID = "grid"
COLUMN_GRID_NAMED = "Netzanschluss"
COLUMN_NET_IMPORT = "Netzbezug"
COLUMN_CONSUMPTION = "consumption"
COLUMN_CONSUMPTION_NAMED = "Brutto Gesamtverbrauch"
COLUMN_BATTERY = "battery"
COLUMN_BATTERY_POS = "battery_pos"
COLUMN_BATTERY_NAMED = "Batterie Durchsatz"
COLUMN_PV = "pv"
COLUMN_PV_PROD = "PV Produktion"
COLUMN_PV_NEG = "pv_neg"
COLUMN_PV_EXCESS = "pv_excess"
COLUMN_PV_FEEDIN = "PV Einspeisung"
COLUMN_PV_SELF = "PV Eigenverbrauch"
COLUMN_PV_BAT = "pv_bat"
COLUMN_PV_IN_BAT = "PV in Batterie"
COLUMN_PV_SHARE = "PV Eigenverbrauchsanteil"
COLUMN_PV_THROUGHPUT = "PV Durchsatz"


def transform_energy_dataframe(
    df: pd.DataFrame,
    component_types: List[str],
    mcfg: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Transform and enrich energy dataframe.

    This function processes a raw DataFrame containing energy metrics,
    adding derived columns for PV production, battery throughput, and grid metrics.

    Args:
        df: Raw DataFrame with energy metrics, expected to have a datetime index.
        component_types: List of component types present in the DataFrame (e.g., ["pv", "battery"]).
        mcfg: Microgrid configuration object providing component type IDs and other metadata.

    Returns:
        A tuple containing:
        - `main_df`: A DataFrame with main columns for visualization and reporting.
        - `df_renamed`: A fully enriched DataFrame.
    """
    # Ensure the DataFrame has a datetime index
    df = df.reset_index()

    # Enrich with PV-related columns
    if "pv" in component_types:
        df[COLUMN_PV_PROD] = -df.get(COLUMN_PV_NEG, 0)
        df[COLUMN_PV_EXCESS] = (df[COLUMN_PV_PROD] - df[COLUMN_CONSUMPTION]).clip(
            lower=0
        )

        if "battery" in component_types:
            df[COLUMN_PV_IN_BAT] = df[[COLUMN_PV_EXCESS, COLUMN_BATTERY_POS]].min(
                axis=1
            )
        else:
            df[COLUMN_PV_IN_BAT] = 0

        df[COLUMN_PV_FEEDIN] = df[COLUMN_PV_EXCESS] - df[COLUMN_PV_IN_BAT]
        df[COLUMN_PV_SELF] = (df[COLUMN_PV_PROD] - df[COLUMN_PV_EXCESS]).clip(lower=0)
        df[COLUMN_PV_SHARE] = df[COLUMN_PV_SELF] / df[COLUMN_CONSUMPTION].replace(
            0, pd.NA
        )

    # Convert timestamp to Berlin time
    if df[COLUMN_TIMESTAMP].dt.tz is None:
        df[COLUMN_TIMESTAMP] = df[COLUMN_TIMESTAMP].dt.tz_localize("UTC")
    df[COLUMN_TIMESTAMP] = df[COLUMN_TIMESTAMP].dt.tz_convert(TZ_NAME)

    # Basic renaming
    rename_map: Dict[str, str] = {
        COLUMN_TIMESTAMP: COLUMN_TIMESTAMP_NAMED,
        COLUMN_GRID: COLUMN_GRID_NAMED,
        COLUMN_CONSUMPTION: COLUMN_CONSUMPTION_NAMED,
    }

    if "battery" in component_types:
        rename_map[COLUMN_BATTERY] = COLUMN_BATTERY_NAMED

    if "pv" in component_types:
        rename_map.update(
            {
                "pv": COLUMN_PV_THROUGHPUT,
                COLUMN_PV_PROD: COLUMN_PV_PROD,
                COLUMN_PV_SELF: COLUMN_PV_SELF,
                COLUMN_PV_FEEDIN: COLUMN_PV_FEEDIN,
                COLUMN_PV_SHARE: COLUMN_PV_SHARE,
            }
        )
        if "battery" in component_types:
            rename_map[COLUMN_PV_BAT] = COLUMN_PV_IN_BAT

    # Rename individual component IDs
    single_comp = [col for col in df.columns if col.isdigit()]

    if "battery" in component_types:
        battery_ids = {
            str(i) for i in mcfg.component_type_ids(component_type="battery")
        }
        rename_map.update(
            {col: f"Batterie #{col}" for col in single_comp if col in battery_ids}
        )

    if "pv" in component_types:
        pv_ids = {str(i) for i in mcfg.component_type_ids(component_type="pv")}
        rename_map.update({col: f"PV #{col}" for col in single_comp if col in pv_ids})

    df_renamed = df.rename(columns=rename_map)

    # Add derived net import column
    df_renamed[COLUMN_NET_IMPORT] = df_renamed[COLUMN_GRID_NAMED].clip(lower=0)

    # Select main columns for compact display
    def _get_main_columns(
        columns: Iterable[str], component_types: List[str]
    ) -> List[str]:
        base = {
            COLUMN_TIMESTAMP_NAMED,
            COLUMN_GRID_NAMED,
            COLUMN_NET_IMPORT,
            COLUMN_CONSUMPTION_NAMED,
        }

        if "battery" in component_types:
            base.add(COLUMN_BATTERY_NAMED)

        if "pv" in component_types:
            base.update(
                {
                    COLUMN_PV_THROUGHPUT,
                    COLUMN_PV_PROD,
                    COLUMN_PV_SELF,
                    COLUMN_PV_FEEDIN,
                }
            )
            if "battery" in component_types:
                base.update({COLUMN_PV_IN_BAT, COLUMN_PV_SHARE})

        # Add individual component columns like "PV #1", "Batterie #3", etc.
        base.update({col for col in columns if "#" in col})

        return [col for col in columns if col in base]

    main_df = df_renamed[_get_main_columns(df_renamed.columns, component_types)]

    return main_df, df_renamed


def compute_power_df(
    main_df: pd.DataFrame, resolution: Union[str, pd.Timedelta]
) -> pd.DataFrame:
    """Compute energy mix (PV vs grid) and return a summary power DataFrame.

    Args:
        main_df: DataFrame with energy data, including 'Netzbezug'
                 and optionally 'PV Eigenverbrauch'.
        resolution: Time resolution of each row in the DataFrame (e.g., "15min").

    Returns:
        A DataFrame summarizing the energy source mix in kWh, %, and average kW.
    """
    resolution = pd.to_timedelta(resolution)
    hours = resolution.total_seconds() / 3600

    # Calculate energy from grid
    grid_kwh = round(main_df[COLUMN_NET_IMPORT].sum() * hours, 2)

    if COLUMN_PV_SELF in main_df.columns:
        # Calculate energy from PV
        pv_self_kwh = round(main_df[COLUMN_PV_SELF].sum() * hours, 2)
        total_kwh = pv_self_kwh + grid_kwh

        energy_kwh = [pv_self_kwh, grid_kwh]
        energy_labels = ["PV", "Netz"]

        return pd.DataFrame(
            {
                "Energiebezug": energy_labels,
                "Energie [kWh]": energy_kwh,
                "Energie %": [round(e / total_kwh * 100, 2) for e in energy_kwh],
                "Energie [kW]": [round(e / hours, 2) for e in energy_kwh],
            }
        )

    # Only grid consumption available
    return pd.DataFrame(
        {
            "Energiebezug": ["Netz"],
            "Energie [kWh]": [grid_kwh],
            "Energie %": [100.0],
            "Energie [kW]": [round(grid_kwh / hours, 2)],
        }
    )


def print_pv_sums(main_df: pd.DataFrame, resolution: pd.Timedelta) -> None:
    """Print formatted sums for each PV column.

    Args:
        main_df: DataFrame containing PV columns with energy data.
        resolution: Time resolution of each row in the DataFrame (e.g., "15min").
    """
    pv_columns = [col for col in main_df.columns.tolist() if "PV #" in col]

    for pv in pv_columns:
        pv_sum = round(main_df[pv].sum() * resolution * -1, 2)
        formatted_sum = (
            f"{pv_sum:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        print(f"{pv:<7}:   {formatted_sum} kWh")


def create_pv_analysis_df(
    main_df: pd.DataFrame,
    pv_filter: List[str],
    pvgrid_filter: str,
    pv_grid_filter_options: List[str],
) -> pd.DataFrame:
    """Create a DataFrame for PV analysis based on selected filters.

    Args:
        main_df: DataFrame containing PV and grid data.
        pv_filter: List of PV components to include (e.g., ["1", "2"] or ["Alle"]).
        pvgrid_filter: Filter option for PV and grid analysis (e.g., "PV", "Grid", "PV + Grid").
        pv_grid_filter_options: List of available filter options for PV and grid.
    Returns:
        A DataFrame with PV feed-in data, reshaped for analysis.
    """
    # Case 1: Only PV
    if pvgrid_filter == pv_grid_filter_options[1]:
        pv_columns = (
            [col for col in main_df.columns if "PV #" in col]
            if "Alle" in pv_filter
            else [f"PV {pv}" for pv in pv_filter]
        )
        df = main_df[[COLUMN_TIMESTAMP_NAMED] + pv_columns].copy()
        df = df.melt(
            id_vars=[COLUMN_TIMESTAMP_NAMED],
            value_vars=pv_columns,
            var_name="PV",
            value_name=COLUMN_PV_FEEDIN,
        )
        df[COLUMN_PV_FEEDIN] *= -1
        df["PV"] = df["PV"].str[3:]

    # Case 2: Only Grid
    elif pvgrid_filter == pv_grid_filter_options[2]:
        df = main_df[[COLUMN_TIMESTAMP_NAMED, COLUMN_GRID_NAMED]].copy()
        df["PV"] = "#"

    # Case 3: Grid + PV split
    else:
        pv_columns = (
            [col for col in main_df.columns if "PV #" in col]
            if "Alle" in pv_filter
            else [f"PV {pv}" for pv in pv_filter]
        )
        df = main_df[[COLUMN_TIMESTAMP_NAMED, COLUMN_GRID_NAMED] + pv_columns].copy()
        df = df.melt(
            id_vars=[COLUMN_TIMESTAMP_NAMED, COLUMN_GRID_NAMED],
            value_vars=pv_columns,
            var_name="PV",
            value_name=COLUMN_PV_FEEDIN,
        )
        df[COLUMN_GRID_NAMED] /= len(pv_columns)
        df[COLUMN_PV_FEEDIN] *= -1
        df["PV"] = df["PV"].str[3:]

    return df


def create_battery_analysis_df(
    main_df: pd.DataFrame, bat_filter: List[str]
) -> pd.DataFrame:
    """Create a DataFrame for battery analysis based on selected filters.

    Args:
        main_df: DataFrame containing battery data.
        bat_filter: List of battery components to include (e.g., ["1", "2"] or ["Alle"]).
    Returns:
        A DataFrame with battery throughput data, reshaped for analysis.
    """
    bat_columns = (
        [col for col in main_df.columns if "Batterie #" in col]
        if "Alle" in bat_filter
        else [f"Batterie {i}" for i in bat_filter]
    )

    df = main_df[bat_columns].copy()
    df[COLUMN_TIMESTAMP_NAMED] = main_df.index

    df = df.melt(
        id_vars=[COLUMN_TIMESTAMP_NAMED],
        value_vars=bat_columns,
        var_name="Batterie",
        value_name=COLUMN_BATTERY_NAMED,
    )
    df["Batterie"] = df["Batterie"].str[9:]

    return df
