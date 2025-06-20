# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Data processing functions for the reporting module.

Overview:
---------
This module contains a series of data transformation functions used to generate
energy reports from microgrid component data, such as PV (photovoltaic), battery,
and grid consumption metrics. These functions are typically executed in a specific
order within the reporting notebook. The output of one function is often used as
input to the next, forming a processing pipeline.

The functions handle:
- Timezone normalization
- Data enrichment (e.g., PV metrics, grid net usage)
- Column renaming based on component configuration
- Aggregation and summarization of energy data
- Generation of overview tables and analysis-ready DataFrames

Assumptions and Requirements:
-----------------------------
- Input `df` must contain at least the columns: `"timestamp"`, `"grid"`
- Additional columns like `"pv_neg"` and `"battery_pos"` are required
  for PV and battery metrics.
- Timestamps must be in datetime format; timezone-naive timestamps
  are assumed to be in UTC.
- Component configuration `mcfg` must implement
  `component_type_ids(...)` returning a list of IDs.
- `component_types` is a list containing any of: `"grid"`, `"consumption"`,
  `"pv"`, `"battery"`, `"chp"`, `"ev"`.

Output:
-------
Most functions return either:
- A modified `pd.DataFrame` with new or renamed columns,
- A summary `dict` of computed statistics,
- Or a reshaped long-format DataFrame for visual inspection or plotting.

This modular pipeline ensures flexibility while maintaining clear structure
for preparing reproducible, component-aware energy reporting.
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def _get_rename_map(component_types: List[str]) -> Dict[str, str]:
    """Return a mapping from raw column names to human-readable German names."""
    rename_map: Dict[str, str] = {
        "timestamp": "Zeitpunkt",
        "grid": "Netzanschluss",
        "consumption": "Brutto Gesamtverbrauch",
    }

    if "battery" in component_types:
        rename_map["battery"] = "Batterie Durchsatz"

    if "pv" in component_types:
        rename_map.update(
            {
                "pv": "PV Durchsatz",
                "pv_prod": "PV Produktion",
                "pv_self": "PV Eigenverbrauch",
                "pv_bat": "PV in Batterie",
                "pv_feedin": "PV Einspeisung",
                "pv_self_consumption_share": "PV Eigenverbrauchsanteil",
            }
        )

    return rename_map


def convert_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'timestamp' column to Europe/Berlin timezone."""
    assert "timestamp" in df.columns, df
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Berlin")
    return df


def process_grid_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'Netzbezug' column for positive grid consumption."""
    df["Netzbezug"] = df["Netzanschluss"].clip(lower=0)
    return df


def compute_pv_metrics(df: pd.DataFrame, component_types: List[str]) -> pd.DataFrame:
    """Compute PV-related metrics and add them to the DataFrame."""
    df["pv_prod"] = -df["pv_neg"]
    df["pv_excess"] = (df["pv_prod"] - df["consumption"]).clip(lower=0)
    if "battery" in component_types:
        df["pv_bat"] = df[["pv_excess", "battery_pos"]].min(axis=1)
    else:
        df["pv_bat"] = 0
    df["pv_feedin"] = df["pv_excess"] - df["pv_bat"]
    df["pv_self"] = (df["pv_prod"] - df["pv_excess"]).clip(lower=0)
    df["pv_self_consumption_share"] = df["pv_self"] / df["consumption"].replace(
        0, pd.NA
    )
    return df


def apply_renaming(
    df: pd.DataFrame, component_types: List[str], mcfg: Any
) -> pd.DataFrame:
    """Apply full renaming: static columns and dynamic component columns."""
    # Step 1: Static column renaming
    rename_map = _get_rename_map(component_types)

    # Step 2: Dynamic component column renaming
    single_comp = [col for col in df.columns if col.isdigit()]
    if "battery" in component_types:
        battery_ids = {
            str(i)
            for i in mcfg.component_type_ids(
                component_type="battery", component_category="meter"
            )
        }
        rename_map.update(
            {col: f"Batterie #{col}" for col in single_comp if col in battery_ids}
        )
    if "pv" in component_types:
        pv_ids = {
            str(i)
            for i in mcfg.component_type_ids(
                component_type="pv", component_category="meter"
            )
        }
        rename_map.update({col: f"PV #{col}" for col in single_comp if col in pv_ids})

    return df.rename(columns=rename_map)


def prepare_reporting_dfs(
    df: pd.DataFrame, component_types: List[str], mcfg: Any
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create master and renamed DataFrames based on component types and config."""
    df = df.reset_index()
    df = convert_timezone(df)

    df_renamed = apply_renaming(df, component_types, mcfg)
    df_renamed = process_grid_data(df_renamed)

    master_df = df_renamed[_get_master_columns(df_renamed.columns, component_types)]
    return master_df, df_renamed


def _get_master_columns(
    columns: pd.Index[str], component_types: List[str]
) -> List[str]:
    """Determine relevant columns for the master DataFrame based on component types."""
    cols = ["Zeitpunkt", "Netzanschluss", "Netzbezug", "Brutto Gesamtverbrauch"]

    if "battery" in component_types:
        cols.append("Batterie Durchsatz")

    if "pv" in component_types:
        cols += [
            "PV Durchsatz",
            "PV Produktion",
            "PV Eigenverbrauch",
            "PV Einspeisung",
        ]
        if "battery" in component_types:
            cols += ["PV in Batterie", "PV Eigenverbrauchsanteil"]

    # Add individual components like "PV#1", "Battery#3", etc.
    cols += [col for col in columns if "#" in col]

    return cols


def create_overview_df(
    master_df: pd.DataFrame, component_types: List[str]
) -> pd.DataFrame:
    """Create an overview dataframe with selected columns based on component types."""
    if "pv" in component_types and "battery" in component_types:
        return master_df[
            [
                "Zeitpunkt",
                "Netzbezug",
                "Brutto Gesamtverbrauch",
                "PV Produktion",
                "PV Einspeisung",
                "Batterie Durchsatz",
            ]
        ]
    if "battery" in component_types:
        return master_df[
            ["Zeitpunkt", "Netzbezug", "Brutto Gesamtverbrauch", "Batterie Durchsatz"]
        ]
    if "pv" in component_types:
        return master_df[
            [
                "Zeitpunkt",
                "Netzbezug",
                "Brutto Gesamtverbrauch",
                "PV Produktion",
                "PV Einspeisung",
            ]
        ]
    return master_df[["Zeitpunkt", "Netzbezug", "Brutto Gesamtverbrauch"]]


def compute_power_df(
    master_df: pd.DataFrame, resolution: Union[str, pd.Timedelta]
) -> pd.DataFrame:
    """Compute energy mix (PV vs grid) and return power dataframe."""
    resolution = pd.to_timedelta(resolution)
    hours = resolution.total_seconds() / 3600
    grid_kwh = round(master_df["Netzbezug"].sum() * hours, 2)
    if "PV Eigenverbrauch" in master_df.columns:
        pv_self_kwh = round(master_df["PV Eigenverbrauch"].sum() * hours, 2)
        total = pv_self_kwh + grid_kwh
        energy = [pv_self_kwh, grid_kwh]
        return pd.DataFrame(
            {
                "Energiebezug": ["PV", "Netz"],
                "Energie [kWh]": energy,
                "Energie %": [round(e / total * 100, 2) for e in energy],
                "Energie [kW]": [
                    round(e * 3600 / resolution.total_seconds(), 2) for e in energy
                ],
            }
        )
    return pd.DataFrame(
        {
            "Energiebezug": ["Netz"],
            "Energie [kWh]": [grid_kwh],
            "Energie %": [100.0],
            "Energie [kW]": [round(grid_kwh * 3600 / resolution.total_seconds(), 2)],
        }
    )


def compute_pv_statistics(
    master_df: pd.DataFrame, component_types: List[str], resolution: pd.Timedelta
) -> Dict[str, Union[int, float]]:
    """Compute PV-related statistics."""
    hours = resolution.total_seconds() / 3600
    stats: Dict[str, float] = {
        "pv_feed_in_sum": 0.0,
        "pv_production_sum": 0.0,
        "pv_self_consumption_sum": 0.0,
        "pv_bat_sum": 0.0,
        "pv_self_consumption_share": 0.0,
        "pv_total_consumption_share": 0.0,
    }
    if "pv" not in component_types:
        return stats
    pv_prod = master_df.get("PV Produktion", pd.Series(dtype=float))
    if pv_prod.sum() <= 0:
        return stats
    stats["pv_feed_in_sum"] = round((master_df["PV Einspeisung"] * hours).sum(), 2)
    stats["pv_production_sum"] = round((pv_prod * hours).sum(), 2)
    stats["pv_self_consumption_sum"] = round(
        (master_df["PV Eigenverbrauch"] * hours).sum(), 2
    )
    if "battery" in component_types:
        stats["pv_bat_sum"] = round((master_df["PV in Batterie"] * hours).sum(), 2)
    if stats["pv_production_sum"] > 0:
        stats["pv_self_consumption_share"] = round(
            stats["pv_self_consumption_sum"] / stats["pv_production_sum"], 4
        )
    total_consumed = stats["pv_self_consumption_sum"] + round(
        master_df["Netzbezug"].sum() * hours, 2
    )
    if total_consumed > 0:
        stats["pv_total_consumption_share"] = round(
            stats["pv_self_consumption_sum"] / total_consumed, 4
        )
    return stats


def compute_peak_usage(
    master_df: pd.DataFrame, resolution: pd.Timedelta
) -> Dict[str, Union[str, float]]:
    """Get peak grid usage, corresponding date, and net site consumption sum."""
    peak = round(master_df["Netzbezug"].max(), 2)
    peak_row = master_df.loc[master_df["Netzbezug"].idxmax()]
    timestamp = peak_row["Zeitpunkt"]
    if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
        peak_date_str = (
            timestamp.astimezone(ZoneInfo("CET")).date().strftime("%d.%m.%Y")
        )
    else:
        peak_date_str = timestamp.strftime("%d.%m.%Y")  # fallback
    hours = resolution.total_seconds() / 3600
    return {
        "peak": peak,
        "peak_date": peak_date_str,
        "net_site_consumption_sum": round(
            master_df["Brutto Gesamtverbrauch"].sum() * hours, 2
        ),
        "grid_consumption_sum": round(master_df["Netzbezug"].sum() * hours, 2),
    }


def filter_overview_df(
    overview_df: pd.DataFrame, overview_filter: pd.DataFrame
) -> pd.DataFrame:
    """Filter overview dataframe based on selected columns."""
    if "Alle" not in overview_filter:
        filtered_df = overview_df.copy()
        for column in overview_df.columns:
            display_name = "Gesamtverbrauch" if column == "Netzbezug" else column
            if display_name not in overview_filter and column != "Zeitpunkt":
                filtered_df[column] = np.nan
    return filtered_df


def print_pv_sums(
    master_df: pd.DataFrame, resolution: pd.Timedelta, pv_columns: List[str]
) -> None:
    """Print formatted sums for each PV column."""
    for pv in pv_columns:
        pv_sum = round(
            master_df[pv].sum() * (resolution.total_seconds() / 3600) * -1, 2
        )
        formatted_sum = (
            f"{pv_sum:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        print(f"{pv:<7}:   {formatted_sum} kWh")


def create_pv_analyse_df(
    master_df: pd.DataFrame,
    pv_filter: List[str],
    pvgrid_filter: str,
    pv_grid_filter_options: List[str],
) -> pd.DataFrame:
    """Create a DataFrame for PV analysis based on selected filters."""
    if pvgrid_filter == pv_grid_filter_options[1]:
        pv_columns = (
            [col for col in master_df.columns if "PV #" in col]
            if "Alle" in pv_filter
            else [f"PV {pv}" for pv in pv_filter]
        )
        df = master_df[["Zeitpunkt"] + pv_columns].copy()
        df = pd.melt(
            df,
            id_vars=["Zeitpunkt"],
            value_vars=pv_columns,
            var_name="PV",
            value_name="PV Einspeisung",
        )
        df["PV Einspeisung"] *= -1
        df["PV"] = df["PV"].str[3:]

    elif pvgrid_filter == pv_grid_filter_options[2]:
        df = master_df[["Zeitpunkt", "Netzanschluss"]].copy()
        df["PV"] = "#"

    else:
        pv_columns = (
            [col for col in master_df.columns if "PV #" in col]
            if "Alle" in pv_filter
            else [f"PV {pv}" for pv in pv_filter]
        )
        df = master_df[["Zeitpunkt"] + pv_columns + ["Netzanschluss"]].copy()
        df = pd.melt(
            df,
            id_vars=["Zeitpunkt", "Netzanschluss"],
            value_vars=pv_columns,
            var_name="PV",
            value_name="PV Einspeisung",
        )
        df["Netzanschluss"] /= len(pv_columns)
        df["PV Einspeisung"] *= -1
        df["PV"] = df["PV"].str[3:]

    return df


def create_battery_analyse_df(master_df: pd.DataFrame, bat_filter: str) -> pd.DataFrame:
    """Create a DataFrame for battery analysis based on selected filters."""
    bat_columns = (
        [col for col in master_df.columns if "Batterie #" in col]
        if "Alle" in bat_filter
        else [f"Batterie {i}" for i in bat_filter]
    )
    df = master_df[bat_columns].copy()
    df["Zeitpunkt"] = df.index
    df = pd.melt(
        df,
        id_vars=["Zeitpunkt"],
        value_vars=bat_columns,
        var_name="Batterie",
        value_name="Batterie Durchsatz",
    )
    df["Batterie"] = df["Batterie"].str[9:]

    return df
