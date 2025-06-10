# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Reporting for asset optimization."""


from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from frequenz.data.microgrid import MicrogridData


# pylint: disable=too-many-arguments
async def fetch_data(
    mdata: MicrogridData,
    *,
    component_types: tuple[str],
    mid: int,
    start_time: datetime,
    end_time: datetime,
    resampling_period: timedelta,
    splits: bool = False,
    fetch_soc: bool = False,
) -> pd.DataFrame:
    """
    Fetch data of a microgrid and processes it for plotting.

    Args:
        mdata: MicrogridData object to fetch data from.
        component_types: List of component types to fetch data for.
        mid: Microgrid ID.
        start_time: Start time for data fetching.
        end_time: End time for data fetching.
        resampling_period: Time resolution for data fetching.
        splits: Whether to split the data into positive and negative parts.
        fetch_soc: Whether to fetch state of charge (SOC) data.

    Returns:
        pd.DataFrame: DataFrame containing the processed data.

    Raises:
        ValueError: If no data is found for the given microgrid and time range or if
            unexpected component types are present in the data.
    """
    print(
        f"Requesting data from {start_time} to {end_time} at {resampling_period} resolution"
    )
    df = await mdata.ac_active_power(
        microgrid_id=mid,
        component_types=component_types,
        start=start_time,
        end=end_time,
        resampling_period=resampling_period,
        keep_components=False,
        splits=splits,
        unit="kW",
    )
    if df is None or df.empty:
        raise ValueError(
            f"No data found for microgrid {mid} between {start_time} and {end_time}"
        )

    print(f"Received {df.shape[0]} rows and {df.shape[1]} columns")

    # For later vizualization we default to zero
    df["battery"] = df.get("battery", 0)
    df["chp"] = df["chp"].clip(upper=0) if "chp" in df.columns else 0
    df["pv"] = df["pv"].clip(upper=0) if "pv" in df.columns else 0

    # Determine consumption if not present
    if "consumption" not in df.columns:
        if any(
            ct not in ["grid", "pv", "battery", "chp", "consumption"]
            for ct in df.columns
        ):
            raise ValueError(
                f"Consumption not found in data and unexpected component types present: {df.columns.tolist()}"
            )
        df["consumption"] = df["grid"] - (df["chp"] + df["pv"] + df["battery"])

    if fetch_soc:
        soc_df = await mdata.soc(
            microgrid_id=mid,
            start=start_time,
            end=end_time,
            resampling_period=resampling_period,
            keep_components=False,
        )
        if soc_df is None or soc_df.empty:
            raise ValueError(
                f"No SOC data found for microgrid {mid} between {start_time} and {end_time}"
            )
        df = pd.concat([df, soc_df.rename(columns={"battery": "soc"})[["soc"]]], axis=1)

    df["soc"] = df.get("soc", np.nan)

    return df


def plot_power_flow(df: pd.DataFrame, ax: Axes | None = None) -> None:
    """Plot the power flow of the microgrid."""
    d = -df.copy()
    i = d.index
    cons = -d["consumption"].to_numpy()

    has_chp = "chp" in d.columns
    has_pv = "pv" in d.columns
    chp = d["chp"] if has_chp else 0 * cons
    prod = chp + (d["pv"].clip(lower=0) if has_pv else 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 10), sharex=True)

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
        fig, ax = plt.subplots(figsize=(30, 10), sharex=True)
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
        2, 1, figsize=(30, 10), sharex=True, height_ratios=[4, 1]
    )
    plot_power_flow(df, ax=ax1)
    plot_energy_trade(df, ax=ax2)
    plt.tight_layout()
    plt.show()


def plot_battery_power(df: pd.DataFrame) -> None:
    """Plot the battery power and state of charge (SOC) of the microgrid."""
    if "soc" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'soc' column for battery SOC plotting."
        )

    fig, ax1 = plt.subplots(figsize=(30, 6.66))  # Increased the figure height

    # Plot Battery SOC
    twin_ax = ax1.twinx()
    assert df["soc"].ndim == 1, "SOC data should be 1D"
    soc = df["soc"]  # .iloc[:, 0] if df["soc"].ndim > 1 else df["soc"]
    twin_ax.grid(False)  # Turn off the grid for the SOC plot
    twin_ax.fill_between(
        df.index,
        soc.to_numpy() * 0,
        soc.to_numpy(),
        color="grey",
        alpha=0.4,
        label="SOC",
    )
    twin_ax.set_ylim(0, 100)  # Set SOC range between 0 and 100
    twin_ax.set_ylabel("Battery SOC", fontsize=14)  # Increased the font size
    twin_ax.tick_params(
        axis="y", labelcolor="grey", labelsize=14
    )  # Increased the font size for ticks

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
    ax1.set_ylabel(
        "Battery Power", fontsize=14
    )  # Updated the label to include Grid - Bat
    ax1.tick_params(
        axis="y", labelcolor="black", labelsize=14
    )  # Increased the font size for ticks

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
    fig.legend(loc="upper left", fontsize=14)  # Increased font size for legend
    plt.show()


def plot_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Plot monthly aggregate data."""
    months = df.resample("1MS").sum()
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
