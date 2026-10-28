# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Energy reporting metrics for production, consumption, and storage analysis.

This module provides helper functions for calculating and analyzing directional
flows within a hybrid power system — including production, consumption,
battery charging, grid feed-in, and self-consumption metrics.

Key features:
It standardizes sign conventions (production is negative in PSC) and ensures
consistency in derived quantities such as:
  - Production surplus (excess generation)
  - Energy stored in a battery
  - Grid feed-in power
  - Self-consumption and self-consumption share
  - Inferred consumption when not explicitly provided

Usage:
    These functions serve as building blocks for energy reporting and
    dashboards that report on the performance of energy assets.
"""

import warnings
from datetime import timedelta
from typing import Literal

import pandas as pd

GridDataSource = Literal["power", "energy"]
DEFAULT_RESOLUTION = timedelta(seconds=10)


def grid_directions(
    *,
    data_source: GridDataSource,
    grid: pd.Series | None = None,
    grid_import: pd.Series | None = None,
    grid_export: pd.Series | None = None,
    resolution: timedelta | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Normalize grid power or energy deltas into directional power series.

    Args:
        data_source: "power" for one signed PSC grid series or "energy"
            for separate consumed and delivered energy-delta series.
        grid: Signed grid power (positive import, negative export).
        grid_import: Directional grid-import values.
        grid_export: Directional grid-export values.
        resolution: Reporting period duration. Required for energy input so the
            energy delta can be converted to average power.

    Returns:
        Grid consumption and grid feed-in as non-negative power series.

    Raises:
        ValueError: If the selected source's inputs are missing or invalid.
    """
    if data_source == "power":
        if grid is None:
            raise ValueError("grid must be provided when data_source='power'.")
        grid_power = grid.astype("float64")
        return grid_power.clip(lower=0), grid_power.clip(upper=0).abs()

    if data_source != "energy":
        raise ValueError(f"Unknown grid data source: {data_source}")
    if grid_import is None or grid_export is None:
        raise ValueError(
            "grid_import and grid_export must be provided when data_source='energy'."
        )
    if resolution is None or resolution <= timedelta(0):
        raise ValueError(
            "A positive resolution must be provided when data_source='energy'."
        )

    power_factor = timedelta(hours=1) / resolution
    return (
        grid_import.astype("float64").clip(lower=0) * power_factor,
        grid_export.astype("float64").clip(lower=0) * power_factor,
    )


def asset_production(production: pd.Series) -> pd.Series:
    """Extract the positive production portion from a directional series.

    Assumes PSC convention (production negative) and inverts to keep the
    productive (non-negative) portion.

    Args:
        production: Production values using either power or energy units.

    Returns:
        A Series where only production values (≥ 0) are retained, with all
        non-productive values set to zero. Missing values remain NaN.
    """
    return (-production).clip(lower=0)


def production_excess(production: pd.Series, consumption: pd.Series) -> pd.Series:
    """Compute the excess production relative to consumption.

    Calculates surplus by subtracting consumption from production and removing
    negative results. Production is optionally sign-corrected first.

    Args:
        production: Series of production values (PSC convention, negative).
        consumption: Series of consumption values (same units as `production`).

    Returns:
        A Series representing excess production (≥ 0).
    """
    asset_production_series = asset_production(production)
    return (asset_production_series - consumption).clip(lower=0)


def grid_feed_in(
    grid_export: pd.Series | None,
    production: pd.Series | None,
    consumption: pd.Series | None,
    battery: pd.Series | None,
) -> pd.Series:
    """Determine grid feed-in using measured or derived series.

    Prefers measured grid export. When unavailable, derives grid feed-in from
    the remaining balance. All provided series must represent the same
    measurement kind and unit (for example, kW or kWh).

    Args:
        grid_export: Measured grid-export series (positive), or None.
        production: Production series (PSC convention), or None.
        consumption: Consumption series (PSC convention), or None.
        battery: Battery series (PSC convention: charging positive), or None.

    Returns:
        A Series representing power or energy fed into the grid (≥ 0).

    Raises:
        ValueError: If no valid series are provided.
    """
    if grid_export is not None:
        return grid_export.astype("float64").clip(lower=0)

    if all(s is None for s in (production, consumption, battery)):
        raise ValueError(
            "Cannot compute grid_feed_in because no grid export, production, "
            "consumption, or battery series were provided. "
            "This function follows PSC conventions, so production and battery "
            "must also follow those conventions."
        )

    prod = production.astype("float64") if production is not None else 0.0
    cons = consumption.astype("float64") if consumption is not None else 0.0
    batt = battery.astype("float64") if battery is not None else 0.0

    inferred = cons + prod + batt

    return inferred.clip(upper=0).abs()  # type: ignore[union-attr]


def production_self_consumption(
    production: pd.Series, consumption: pd.Series
) -> pd.Series:
    """Compute the portion of production directly self-consumed.

    Calculates the part of total production that is used locally rather than
    stored or exported, by subtracting excess production from total production.

    Args:
        production: Series of production values (e.g., kW or MW).
        consumption: Series of consumption values (same units as `production`).

    Returns:
        A Series representing self-consumed production.

    Warns:
        UserWarning: If negative self-consumption values are detected, indicating
            that the computed excess exceeds total production for some entries.
    """
    asset_production_series = asset_production(production)
    production_excess_series = production_excess(production, consumption)
    result = asset_production_series - production_excess_series

    if (result < 0).any():
        warnings.warn(
            "Negative self-consumption values detected. "
            "This indicates production excess exceeds total production for some entries.",
            UserWarning,
            stacklevel=2,
        )

    return result


def production_self_usage(production: pd.Series, consumption: pd.Series) -> pd.Series:
    """Calculate the self-consumption share of total consumption.

    Computes the ratio of self-used production to total consumption,
    representing how much of the consumed energy was covered by own production.

    Args:
        production: Series of production values (e.g., kW or MW).
        consumption: Series of consumption values (same units as `production`).

    Returns:
        A Series expressing the self-consumption share (values between 0 and 1).
        Returns NaN where consumption is zero.
    """
    production_self_use = production_self_consumption(production, consumption)
    denom = consumption.astype("float64")
    denom = denom.mask(denom <= 0)  # NaN when consumption <= 0
    share = production_self_use.astype("float64") / denom
    return share


def production_self_share(production: pd.Series, consumption: pd.Series) -> pd.Series:
    """Compute the self-consumption share relative to total production.

    Calculates the fraction of total production that is directly self-consumed.
    The denominator is the positive production portion, and values are masked
    to NaN where production is zero or negative to avoid invalid divisions.

    Args:
        production: Production values using either power or energy units.
        consumption: Consumption values using the same measurement kind and
            unit as production.
    Returns:
        Series of self-consumption share values (typically between 0 and 1).
        Returns NaN for timestamps where total production is zero or negative.
    """
    production_self_use = production_self_consumption(production, consumption)
    denom = asset_production(production).astype("float64")
    denom = denom.mask(denom <= 0)  # NaN when production <= 0
    return production_self_use.astype("float64") / denom


def consumption(
    grid_import: pd.Series,
    grid_export: pd.Series,
    production: pd.Series | None = None,
    battery: pd.Series | None = None,
) -> pd.Series:
    """Infer consumption from directional grid, production, and battery series.

    Computes: consumption = import - export - production - battery. All inputs
    must represent the same measurement kind and unit.

    Args:
        grid_import: Positive grid-import series.
        grid_export: Positive grid-export series.
        production: Optional Series of on-site production values.
            If None, production is treated as zero.
        battery: Optional battery series. Positive values represent charging;
            negative values represent discharging. If None, the battery
            contribution is treated as zero.

    Returns:
        A Series representing inferred total consumption, named `"consumption"`.

    Raises:
        ValueError: If either directional grid series is None.

    """
    if grid_import is None or grid_export is None:
        raise ValueError("Both directional grid series must be provided.")

    grid_s = grid_import.astype("float64") - grid_export.astype("float64")

    # Ensure raw production values are used (usually negative for production)
    if production is None:
        prod_s = pd.Series(0.0, index=grid_s.index)
    else:
        prod_s = production.astype("float64")

    if battery is None:
        battery_s = pd.Series(0.0, index=grid_s.index)
    else:
        battery_s = battery.astype("float64")

    result = (grid_s - prod_s - battery_s).astype("float64")
    result.name = "consumption"

    return result


def grid_consumption(
    grid_import: pd.Series | None,
    production: pd.Series | None,
    consumption: pd.Series | None,
    battery: pd.Series | None,
) -> pd.Series:
    """Determine grid import using measured or derived series.

    This function follows the PSC (Production-Storage-Consumption) convention:
    - Production is negative when generating.
    - Battery is positive when charging (consuming energy).
    - Consumption is positive when consuming energy.

    Prefers measured grid import. When unavailable, derives it from the
    remaining balance. All provided series must represent the same measurement
    kind and unit.

    Args:
        grid_import: Measured grid-import series (positive), or None.
        production: Production series (PSC convention), or None.
        consumption: Consumption series (PSC convention), or None.
        battery: Battery series (PSC convention: charging positive), or None.

    Returns:
        Series with non-negative grid import values.

    Raises:
        ValueError: If neither a grid series nor any of production,
            consumption, or battery series are provided.
    """
    if grid_import is not None:
        return grid_import.astype("float64").clip(lower=0)

    # Ensure at least one other input is provided
    if all(s is None for s in (production, consumption, battery)):
        raise ValueError(
            "Cannot compute grid_consumption because no grid import, production, "
            "consumption, or battery series were provided. "
            "This function follows PSC conventions, so production and battery "
            "must also follow those conventions."
        )

    prod = production.astype("float64") if production is not None else 0.0
    cons = consumption.astype("float64") if consumption is not None else 0.0
    batt = battery.astype("float64") if battery is not None else 0.0

    inferred = cons + prod + batt

    # We only want the import portion (≥ 0)
    return inferred.clip(lower=0)  # type: ignore[union-attr]


# pylint: disable=too-many-locals
def battery_power_flows(
    grid_import: pd.Series,
    grid_export: pd.Series,
    production: pd.Series,
    consumption: pd.Series,
    battery: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute battery charge/discharge flow splits from directional series.

    The calculation supports power or energy inputs as long as all series use
    the same measurement kind and unit. Reporting passes power series here.

    Args:
        grid_import: Non-negative grid-import series.
        grid_export: Non-negative grid-export series.
        production: Non-negative total on-site production series.
        consumption: Non-negative gross site consumption series.
        battery: Battery series. Positive values charge the battery;
            negative values discharge the battery.

    Returns:
        A tuple containing production_to_battery, grid_to_battery,
        battery_to_grid, and battery_to_consumption series.
    """
    grid_consumption_series = grid_import.fillna(0.0).astype("float64")
    grid_feed_in_series = grid_export.fillna(0.0).astype("float64")
    production_series = production.fillna(0.0).astype("float64").clip(lower=0)
    consumption_series = consumption.fillna(0.0).astype("float64").clip(lower=0)
    battery_series = battery.fillna(0.0).astype("float64")

    battery_charge_series = battery_series.clip(lower=0)
    battery_discharge_series = (-battery_series.clip(upper=0)).fillna(0.0)

    production_to_battery_series = pd.concat(
        [
            production_excess(production_series * -1, consumption_series),
            battery_charge_series,
        ],
        axis=1,
    ).min(axis=1)
    grid_to_battery_series = (
        pd.concat(
            [
                battery_charge_series - production_to_battery_series,
                grid_consumption_series,
            ],
            axis=1,
        )
        .min(axis=1)
        .clip(lower=0)
    )

    production_self_use_series = production_self_consumption(
        production=production_series * -1,
        consumption=consumption_series,
    )
    load_balance_series = (
        consumption_series - production_self_use_series - grid_consumption_series
    ).clip(lower=0)
    non_exported_discharge_series = (
        battery_discharge_series - grid_feed_in_series
    ).clip(lower=0)
    battery_to_consumption_series = pd.concat(
        [load_balance_series, non_exported_discharge_series], axis=1
    ).max(axis=1)
    battery_to_consumption_series = pd.concat(
        [battery_to_consumption_series, battery_discharge_series], axis=1
    ).min(axis=1)
    battery_to_grid_series = (
        battery_discharge_series - battery_to_consumption_series
    ).clip(lower=0)

    production_to_battery_series.name = "production_to_battery"
    grid_to_battery_series.name = "grid_to_battery"
    battery_to_grid_series.name = "battery_to_grid"
    battery_to_consumption_series.name = "battery_to_consumption"

    return (
        production_to_battery_series,
        grid_to_battery_series,
        battery_to_grid_series,
        battery_to_consumption_series,
    )
