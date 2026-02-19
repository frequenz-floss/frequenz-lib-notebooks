# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Visualization for asset optimization reporting using matplotlib."""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mpl_colors
from matplotlib.axes import Axes

from .viz_colors import (
    AVAILABLE,
    BUY,
    CHARGE,
    CHP,
    CONSUMPTION,
    DISCHARGE,
    GRID,
    PV,
    SELL,
    SOC,
    ZERO_LINE,
)
from .viz_data import (
    prepare_battery_power_data,
    prepare_energy_trade_data,
    prepare_power_flow_data,
)

_logger = logging.getLogger(__name__)


FIGURE_SIZE = (30, 6.66)  # Default figure size for plots


def _mpl_color(color: str) -> str | tuple[float, float, float, float]:
    """Convert shared color strings to matplotlib-friendly formats."""
    if color.startswith("rgba(") and color.endswith(")"):
        parts = color[5:-1].split(",")
        if len(parts) == 4:
            r, g, b, a = (float(p.strip()) for p in parts)
            return (r / 255, g / 255, b / 255, a)
    return mpl_colors.to_rgba(color)


def plot_power_flow(df: pd.DataFrame, ax: Axes | None = None) -> None:
    """Plot the power flow of the microgrid."""
    data = prepare_power_flow_data(df)
    i = data.index
    cons = data.consumption.to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if data.has_pv:
        pv_label = "PV (on CHP)" if data.has_chp else "PV"
        ax.fill_between(
            i,
            data.chp,
            data.production,
            color=_mpl_color(PV),
            alpha=0.7,
            label=pv_label,
        )
    if data.has_chp:
        ax.fill_between(
            i,
            data.chp,
            color=_mpl_color(CHP),
            alpha=0.5,
            label="CHP",
        )

    if data.charge is not None and data.discharge is not None:
        charge_mask = data.charge.notna().tolist()
        discharge_mask = data.discharge.notna().tolist()
        ax.fill_between(
            i,
            cons,
            data.charge,
            where=charge_mask,
            color=_mpl_color(CHARGE),
            alpha=0.2,
            label="Charge",
        )
        ax.fill_between(
            i,
            cons,
            data.discharge,
            where=discharge_mask,
            color=_mpl_color(DISCHARGE),
            alpha=0.5,
            label="Discharge",
        )

    if data.grid is not None:
        ax.plot(i, data.grid, color=_mpl_color(GRID), label="Grid")

    ax.plot(i, cons, color=_mpl_color(CONSUMPTION), label="Consumption")
    ax.set_ylabel("Power [kW]")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))


def plot_energy_trade(df: pd.DataFrame, ax: Axes | None = None) -> None:
    """Plot the energy trade of the microgrid."""
    data = prepare_energy_trade_data(df)

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.fill_between(
        data.index,
        0,
        data.buy.to_numpy(),
        color=_mpl_color(BUY),
        label="Buy",
        step="pre",
    )
    ax.fill_between(
        data.index,
        0,
        data.sell.to_numpy(),
        color=_mpl_color(SELL),
        label="Sell",
        step="pre",
    )
    ax.set_ylabel("Energy [kWh]")
    ax.legend()
    ax.grid(True)


def plot_power_flow_trade(df: pd.DataFrame) -> None:
    """Plot both power flow and energy trade of the microgrid."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=FIGURE_SIZE, sharex=True, height_ratios=[4, 1]
    )
    plot_power_flow(df, ax=ax1)
    plot_energy_trade(df, ax=ax2)
    plt.tight_layout()
    plt.show()


def plot_battery_power(df: pd.DataFrame) -> None:
    """Plot the battery power and state of charge (SOC) of the microgrid."""
    data = prepare_battery_power_data(df)

    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)

    # Plot Battery SOC
    twin_ax = ax1.twinx()
    assert data.soc.ndim == 1, "SOC data should be 1D"
    soc = data.soc
    twin_ax.grid(False)
    twin_ax.fill_between(
        data.index,
        soc.to_numpy() * 0,
        soc.to_numpy(),
        color=_mpl_color(SOC),
        alpha=0.4,
        label="SOC",
    )
    twin_ax.set_ylim(0, 100)
    twin_ax.set_ylabel("Battery SOC", fontsize=14)
    twin_ax.tick_params(axis="y", labelcolor="grey", labelsize=14)

    # Available power
    ax1.plot(
        data.index,
        data.available,
        color=_mpl_color(AVAILABLE),
        linestyle="-",
        label="Available power",
        alpha=1,
    )

    # Plot Battery Power on primary y-axis
    ax1.axhline(y=0, color=_mpl_color(ZERO_LINE), linestyle="--", alpha=0.5)
    # Make battery power range symmetric
    ax1.set_ylim(-data.max_abs_battery * 1.1, data.max_abs_battery * 1.1)
    ax1.set_ylabel("Battery Power", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="black", labelsize=14)

    # Fill Battery Power around zero (reverse sign)
    ax1.fill_between(
        data.index,
        0,
        data.battery,
        where=data.charge.notna().tolist(),
        interpolate=False,
        color=_mpl_color(CHARGE),
        alpha=0.9,
        label="Charge",
    )
    ax1.fill_between(
        data.index,
        0,
        data.battery,
        where=data.discharge.notna().tolist(),
        interpolate=False,
        color=_mpl_color(DISCHARGE),
        alpha=0.9,
        label="Discharge",
    )

    fig.tight_layout()
    fig.legend(loc="upper left", fontsize=14)
    plt.show()


def plot_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Plot monthly aggregate data."""
    months: pd.DataFrame = df.resample("1MS").sum()
    resolution = (df.index[1] - df.index[0]).total_seconds()
    kW2MWh = resolution / 3600 / 1000  # pylint: disable=invalid-name
    months *= kW2MWh
    # Ensure the index is a datetime
    if not isinstance(months.index, pd.DatetimeIndex):
        months.index = pd.to_datetime(months.index)
    months.index = pd.Index(months.index.date)
    pos, neg = (
        months[[c for c in months.columns if "_pos" in c]],
        months[[c for c in months.columns if "_neg" in c]],
    )

    pos = pos.rename(
        columns={
            "grid_pos": "Grid Consumption",
            "battery_pos": "Battery Charge",
            "consumption_pos": "Consumption",
            "pv_pos": "PV Consumption",
            "chp_pos": "CHP Consumption",
        }
    )
    neg = neg.rename(
        columns={
            "grid_neg": "Grid Feed-in",
            "battery_neg": "Battery Discharge",
            "consumption_neg": "Unknown Production",
            "pv_neg": "PV Production",
            "chp_neg": "CHP Production",
        }
    )

    # Remove zero columns
    pos = pos.loc[:, pos.abs().sum(axis=0) > 0]
    neg = neg.loc[:, neg.abs().sum(axis=0) > 0]

    ax = pos.plot.bar()
    neg.plot.bar(ax=ax, alpha=0.7)
    plt.xticks(rotation=0)
    plt.ylabel("Energy [MWh]")
    return months
