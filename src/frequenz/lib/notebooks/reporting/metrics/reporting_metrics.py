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
    return (asset_production_series - consumption.fillna(0)).clip(lower=0)


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
    df: pd.DataFrame, production_cols: list[str] | None, grid_cols: list[str]
) -> pd.Series:
    """Infer the consumption column from grid and production data if missing.

    If a 'consumption' column is not present, it is computed as the total grid import
    (sum of all grid columns) minus total production. Safely handles missing or
    empty production columns by treating them as zero.

    Args:
        df: Input DataFrame containing grid and optional production columns.
        production_cols: List of production column names (e.g., "pv", "chp", "battery" or "ev").
            Can be None or empty if no on-site generation is present.
        grid_cols: List of one or more grid column names.

    Returns:
        A Series representing inferred total consumption, named `"consumption"`.

    Raises:
        ValueError: If `grid_cols` is empty.
    """
    if "consumption" in df.columns:
        return df["consumption"]

    if not grid_cols:
        raise ValueError("At least one grid column must be specified in grid_cols.")

    # Compute total grid import and total production
    grid_total = df[grid_cols].sum(axis=1)

    # Handle empty production columns safely
    if production_cols:
        production_total = df[production_cols].sum(axis=1)
    else:
        # No production → production_total = 0
        production_total = pd.Series(0, index=df.index)

    # Compute inferred consumption (Series)
    consumption = grid_total - production_total
    consumption.name = "consumption"

    return consumption
