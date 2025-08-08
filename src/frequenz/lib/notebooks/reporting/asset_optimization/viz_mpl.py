# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Visualization for asset optimization reporting using matplotlib."""


import logging

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

_logger = logging.getLogger(__name__)


FIGURE_SIZE = (30, 6.66)  # Default figure size for plots


def require_columns(df: pd.DataFrame, *columns: str) -> None:
    """Ensure that the DataFrame contains the required columns."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")


def plot_power_flow(df: pd.DataFrame, ax: Axes | None = None) -> None:
    """Plot the power flow of the microgrid."""
    require_columns(df, "consumption", "battery", "grid")
    d = -df.copy()
    i = d.index
    cons = -d["consumption"].to_numpy()

    has_chp = "chp" in d.columns
    has_pv = "pv" in d.columns
    chp = d["chp"] if has_chp else 0 * cons
    prod = chp + (d["pv"].clip(lower=0) if has_pv else 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if has_pv:
        ax.fill_between(
            i,
            chp,
            prod,
            color="gold",
            alpha=0.7,
            label="PV" + (" (on CHP)" if has_chp else ""),
        )
    if has_chp:
        ax.fill_between(i, chp, color="cornflowerblue", alpha=0.5, label="CHP")

    if "battery" in d.columns:
        bat_cons = -(d["consumption"].to_numpy() + d["battery"].to_numpy())
        charge = bat_cons > cons
        discharge = bat_cons < cons
        ax.fill_between(
            i,
            cons,
            bat_cons,
            where=charge,
            color="green",
            alpha=0.2,
            label="Charge",
        )
        ax.fill_between(
            i,
            cons,
            bat_cons,
            where=discharge,
            color="lightcoral",
            alpha=0.5,
            label="Discharge",
        )

    if "grid" in d.columns:
        ax.plot(i, -d["grid"], color="grey", label="Grid")

    ax.plot(i, cons, "k-", label="Consumption")
    ax.set_ylabel("Power [kW]")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))


def plot_energy_trade(df: pd.DataFrame, ax: Axes | None = None) -> None:
    """Plot the energy trade of the microgrid."""
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

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.fill_between(
        g.index, 0, g.clip(lower=0).to_numpy(), color="darkred", label="Buy", step="pre"
    )
    ax.fill_between(
        g.index,
        0,
        g.clip(upper=0).to_numpy(),
        color="darkgreen",
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
    require_columns(df, "battery", "grid", "soc")

    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)

    # Plot Battery SOC
    twin_ax = ax1.twinx()
    assert df["soc"].ndim == 1, "SOC data should be 1D"
    soc = df["soc"]
    twin_ax.grid(False)
    twin_ax.fill_between(
        df.index,
        soc.to_numpy() * 0,
        soc.to_numpy(),
        color="grey",
        alpha=0.4,
        label="SOC",
    )
    twin_ax.set_ylim(0, 100)
    twin_ax.set_ylabel("Battery SOC", fontsize=14)
    twin_ax.tick_params(axis="y", labelcolor="grey", labelsize=14)

    # Available power
    available = df["battery"] - df["grid"]
    ax1.plot(
        df.index,
        available,
        color="black",
        linestyle="-",
        label="Available power",
        alpha=1,
    )

    # Plot Battery Power on primary y-axis
    ax1.axhline(y=0, color="grey", linestyle="--", alpha=0.5)
    # Make battery power range symmetric
    max_abs_bat = max(
        abs(df["battery"].min()),
        abs(df["battery"].max()),
        abs(available.min()),
        abs(available.max()),
    )
    ax1.set_ylim(-max_abs_bat * 1.1, max_abs_bat * 1.1)
    ax1.set_ylabel("Battery Power", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="black", labelsize=14)

    # Fill Battery Power around zero (reverse sign)
    ax1.fill_between(
        df.index,
        0,
        df["battery"],
        where=(df["battery"].to_numpy() > 0).tolist(),
        interpolate=False,
        color="green",
        alpha=0.9,
        label="Charge",
    )
    ax1.fill_between(
        df.index,
        0,
        df["battery"],
        where=(df["battery"].to_numpy() <= 0).tolist(),
        interpolate=False,
        color="red",
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
