# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Energy reporting metrics for production, consumption, and storage analysis.

This module provides helper functions for calculating and analyzing energy
flows within a hybrid power system — including production, consumption,
battery charging, grid feed-in, and self-consumption metrics.

Key features:
It standardizes sign conventions (via `production_is_positive`) and ensures
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

import pandas as pd


def asset_production(
    production: pd.Series, production_is_positive: bool = False
) -> pd.Series:
    """Extract the positive production portion from a power series.

    Ensures only productive (non-negative) values remain, regardless of the
    sign convention used for production vs. consumption.

    Args:
        production: Series of power values (e.g., kW or MW).
        production_is_positive: Whether production values are already positive.
            If False, `production` is inverted before clipping.

    Returns:
        A Series where only production values (≥ 0) are retained, with all
        non-productive values set to zero.
    """
    return (
        (production if production_is_positive else -production).fillna(0).clip(lower=0)
    )


def production_excess(
    production: pd.Series, consumption: pd.Series, production_is_positive: bool = False
) -> pd.Series:
    """Compute the excess production relative to consumption.

    Calculates surplus by subtracting consumption from production and removing
    negative results. Production is optionally sign-corrected first.

    Args:
        production: Series of production values (e.g., kW or MW).
        consumption: Series of consumption values (same units as `production`).
        production_is_positive: Whether production values are already positive.
            If False, `production` is inverted before clipping.

    Returns:
        A Series representing excess production (≥ 0).
    """
    asset_production_series = asset_production(
        production, production_is_positive=production_is_positive
    )
    return (asset_production_series - consumption).clip(lower=0)


def production_excess_in_bat(
    production: pd.Series,
    consumption: pd.Series,
    battery: pd.Series,
    production_is_positive: bool = False,
) -> pd.Series:
    """Calculate the portion of excess production stored in the battery.

    Compares available production surplus with the battery's charging capability
    at each timestamp and takes the elementwise minimum.

    Args:
        production: Series of production values (e.g., kW or MW).
        consumption: Series of consumption values (same units as `production`).
        battery: Series representing the battery's available charging capacity
            or power limit at each timestamp.
        production_is_positive: Whether production values are already positive.
            If False, `production` is inverted before clipping.

    Returns:
        A Series showing the actual production power stored in the battery.
    """
    production_excess_series = production_excess(
        production, consumption, production_is_positive=production_is_positive
    )
    battery = battery.astype("float64").clip(lower=0)
    return pd.concat([production_excess_series, battery], axis=1).min(axis=1)


def grid_feed_in(
    production: pd.Series,
    consumption: pd.Series,
    battery: pd.Series,
    production_is_positive: bool = False,
) -> pd.Series:
    """Calculate the portion of excess production fed into the grid.

    Subtracts the amount of excess energy stored in the battery from the total
    production surplus to determine how much is exported to the grid.

    Args:
        production: Series of production values (e.g., kW or MW).
        consumption: Series of consumption values (same units as `production`).
        battery: Series representing the battery's available
            charging capacity.
        production_is_positive: Whether production values are already positive. If False,
            `production` is inverted before clipping.

    Returns:
        A Series representing power or energy fed into the grid (≥ 0).
    """
    production_excess_series = production_excess(
        production, consumption, production_is_positive=production_is_positive
    )
    battery_series = production_excess_in_bat(
        production, consumption, battery, production_is_positive=production_is_positive
    )
    return (production_excess_series - battery_series).clip(lower=0)


def production_self_consumption(
    production: pd.Series, consumption: pd.Series, production_is_positive: bool = False
) -> pd.Series:
    """Compute the portion of production directly self-consumed.

    Calculates the part of total production that is used locally rather than
    stored or exported, by subtracting excess production from total production.

    Args:
        production: Series of production values (e.g., kW or MW).
        consumption: Series of consumption values (same units as `production`).
        production_is_positive: Whether production values are already positive.
            If False, `production` is inverted before clipping.

    Returns:
        A Series representing self-consumed production.

    Warns:
        UserWarning: If negative self-consumption values are detected, indicating
            that the computed excess exceeds total production for some entries.
    """
    asset_production_series = asset_production(
        production, production_is_positive=production_is_positive
    )
    production_excess_series = production_excess(
        production, consumption, production_is_positive=production_is_positive
    )
    result = asset_production_series - production_excess_series

    if (result < 0).any():
        warnings.warn(
            "Negative self-consumption values detected. "
            "This indicates production excess exceeds total production for some entries.",
            UserWarning,
            stacklevel=2,
        )

    return result


def production_self_share(
    production: pd.Series, consumption: pd.Series, production_is_positive: bool = False
) -> pd.Series:
    """Calculate the self-consumption share of total consumption.

    Computes the ratio of self-used production to total consumption,
    representing how much of the consumed energy was covered by own production.

    Args:
        production: Series of production values (e.g., kW or MW).
        consumption: Series of consumption values (same units as `production`).
        production_is_positive: Whether production values are already positive.
            If False, `production` is inverted before clipping.

    Returns:
        A Series expressing the self-consumption share (values between 0 and 1).
        Returns NaN where consumption is zero.
    """
    production_self_use = production_self_consumption(
        production, consumption, production_is_positive=production_is_positive
    )
    denom = consumption.astype("float64")
    denom = denom.mask(denom <= 0)  # NaN when consumption <= 0
    share = production_self_use.astype("float64") / denom
    return share


def consumption(
    grid: pd.Series,
    production: pd.Series | None = None,
    battery: pd.Series | None = None,
) -> pd.Series:
    """Infer total consumption from grid power, on-site production, and battery power.

    Computes: consumption = grid_power - production (raw production-neg values)
                            - battery (raw - with positive and negative values).

    Args:
        grid: Series of grid power values (e.g., kW or MW).
        production: Optional Series of on-site production values.
            If None, production is treated as zero.
        battery: Optional Series representing battery discharge/charge power.
            Positive values increase inferred consumption (battery discharge),
            while negative values decrease it (battery charging). If None, the
            battery contribution is treated as zero.

    Returns:
        A Series representing inferred total consumption, named `"consumption"`.

    Raises:
        ValueError: If `grid` is None.

    """
    if grid is None:
        raise ValueError("`grid` must be provided as a pandas Series.")

    grid_s = grid.astype("float64")

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
    grid: pd.Series | None,
    production: pd.Series | None,
    consumption: pd.Series | None,
    battery: pd.Series | None,
) -> pd.Series:
    """Determine grid import using measured or derived series.

    This function follows the PSC (Production-Storage-Consumption) convention:
    - Production is positive when generating.
    - Battery is positive when charging (consuming energy).
    - Consumption is positive when consuming energy.

    Prefers the measured grid series (positive import, negative export). When
    unavailable, derives grid import from the remaining energy balance.

    Args:
        grid: Grid power series (positive import), or ``None``.
        production: Production power series (PSC-convention), or ``None``.
        consumption: Consumption power series (PSC-convention), or ``None``.
        battery: Battery power (PSC-convention: charging positive), or ``None``.

    Returns:
        Series with non-negative grid import values.

    Raises:
        ValueError: If neither a grid series nor any of production,
            consumption, or battery series are provided.
    """
    if grid is not None:
        return grid.astype("float64").clip(lower=0)

    # Ensure at least one other input is provided
    if all(s is None for s in (production, consumption, battery)):
        raise ValueError(
            "Cannot compute grid_consumption because no grid, production, "
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
