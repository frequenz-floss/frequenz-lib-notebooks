# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Data preparation helpers for asset optimization visualizations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class PowerFlowData:
    """Prepared series for power flow plotting."""

    index: pd.Index
    consumption: pd.Series
    chp: pd.Series
    production: pd.Series
    has_chp: bool
    has_pv: bool
    battery_consumption: pd.Series | None
    charge: pd.Series | None
    discharge: pd.Series | None
    grid: pd.Series | None


@dataclass(frozen=True)
class EnergyTradeData:
    """Prepared series for energy trade plotting."""

    index: pd.Index
    buy: pd.Series
    sell: pd.Series


@dataclass(frozen=True)
class BatteryPowerData:
    """Prepared series for battery power plotting."""

    index: pd.Index
    soc: pd.Series
    available: pd.Series
    battery: pd.Series
    charge: pd.Series
    discharge: pd.Series
    max_abs_battery: float


def require_columns(df: pd.DataFrame, *columns: str) -> None:
    """Validate that a DataFrame contains all required columns.

    Checks whether the provided column names exist in the DataFrame. If any
    required columns are missing, a ``ValueError`` is raised listing all
    missing column names.

    Args:
        df:
            The DataFrame to validate.
        *columns:
            One or more column names that must be present in ``df``.

    Raises:
        ValueError:
            If one or more required columns are not found in the DataFrame.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")


def _masked(series: pd.Series, mask: pd.Series) -> pd.Series:
    """Return a series with values masked out using NaNs for plotting gaps."""
    return series.where(mask, other=np.nan)


def prepare_power_flow_data(df: pd.DataFrame) -> PowerFlowData:
    """Prepare normalized power-flow data for plotting.

    Transforms raw microgrid power columns into a structured ``PowerFlowData``
    object suitable for visualization. The function ensures required columns
    exist, applies sign normalization, derives production totals, and splits
    battery behavior into charge and discharge flows.

    Args:
        df:
            Input DataFrame containing at least the columns
            ``"consumption"``, ``"battery"``, and ``"grid"``.
            Optional columns such as ``"chp"`` and ``"pv"`` are used if present.

    Returns:
        A structured container including:
            - normalized consumption series,
            - CHP and PV production signals,
            - combined production values,
            - battery charge and discharge components (if available),
            - grid exchange values,
            - metadata flags indicating available production sources.
    """
    require_columns(df, "consumption", "battery", "grid")
    d = -df.copy()
    cons = -d["consumption"]

    has_chp = "chp" in d.columns
    has_pv = "pv" in d.columns
    chp = d["chp"] if has_chp else pd.Series(0.0, index=cons.index)
    pv = d["pv"].clip(lower=0) if has_pv else pd.Series(0.0, index=cons.index)
    prod = chp + pv

    battery_consumption = None
    charge = None
    discharge = None
    if "battery" in d.columns:
        battery_consumption = -(d["consumption"] + d["battery"])
        charge = _masked(battery_consumption, battery_consumption > cons)
        discharge = _masked(battery_consumption, battery_consumption < cons)

    grid = -d["grid"] if "grid" in d.columns else None

    return PowerFlowData(
        index=d.index,
        consumption=cons,
        chp=chp,
        production=prod,
        has_chp=has_chp,
        has_pv=has_pv,
        battery_consumption=battery_consumption,
        charge=charge,
        discharge=discharge,
        grid=grid,
    )


def prepare_energy_trade_data(df: pd.DataFrame) -> EnergyTradeData:
    """Prepare normalized energy-trade data for plotting.

    Processes raw power time-series data to derive buy and sell energy flows
    relative to site consumption and local production. The function applies
    sign normalization, subtracts on-site generation (PV and CHP if present),
    and converts the resulting power values into resampled energy quantities.

    Args:
        df:
            Input DataFrame containing at least the ``"consumption"`` column.
            Optional columns such as ``"chp"`` and ``"pv"`` are used if available
            to adjust traded energy values.

    Returns:
        A structured container including:
            - resampled timestamp index (15-minute resolution),
            - positive energy purchases (``buy``),
            - negative energy sales (``sell``).
    """
    require_columns(df, "consumption")
    d = -df.copy()
    cons = -d["consumption"]
    trade = cons.copy()

    has_chp = "chp" in d.columns
    has_pv = "pv" in d.columns
    chp = d["chp"] if has_chp else 0 * cons
    prod = chp + (d["pv"].clip(lower=0) if has_pv else 0)
    trade -= prod

    g = trade.resample("15min").mean() / 4

    return EnergyTradeData(
        index=g.index,
        buy=g.clip(lower=0),
        sell=g.clip(upper=0),
    )


def prepare_battery_power_data(df: pd.DataFrame) -> BatteryPowerData:
    """Prepare normalized battery power data for visualization.

    Extracts battery-related signals from a power-flow DataFrame and computes
    derived series used for plotting, including charge, discharge, available
    battery power, and state-of-charge (SOC). The function also determines the
    maximum absolute battery value for consistent axis scaling.

    Args:
        df:
            Input DataFrame containing at least the columns ``"battery"``,
            ``"grid"``, and ``"soc"``. Values are assumed to represent power
            time-series data.

    Returns:
        A structured container including:
            - timestamp index,
            - state of charge (``soc``),
            - available battery power (``available``),
            - raw battery power values,
            - separated charge and discharge series,
            - maximum absolute battery power for plot scaling.
    """
    require_columns(df, "battery", "grid", "soc")

    soc = df["soc"]
    available = df["battery"] - df["grid"]

    max_abs_bat = max(
        abs(df["battery"].min()),
        abs(df["battery"].max()),
        abs(available.min()),
        abs(available.max()),
    )

    charge = _masked(df["battery"], df["battery"] > 0)
    discharge = _masked(df["battery"], df["battery"] <= 0)

    return BatteryPowerData(
        index=df.index,
        soc=soc,
        available=available,
        battery=df["battery"],
        charge=charge,
        discharge=discharge,
        max_abs_battery=max_abs_bat,
    )
