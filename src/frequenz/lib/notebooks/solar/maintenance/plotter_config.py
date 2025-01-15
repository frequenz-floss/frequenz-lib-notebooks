# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
Unified configuration module for plot data preparation and visualisation.

This module contains dataclasses that define the configuration for different
types of plot views. The configurations are designed to be shared between data
preparers and plotters to eliminate redundancy and ensure consistency.

Classes:
    - `BaseViewConfig`: A base class providing common configuration fields.
    - `CalendarViewConfig`: Configuration for calendar view plots.
    - `RollingViewConfig`: Configuration for rolling view plots.
    - `ProfileViewConfig`: Configuration for statistical production profile plots.
    - `DailyViewConfig`: Configuration for daily production plots.
    - `StatsViewConfig`: Configuration for production statistics data.

Features:
    - A `create_config` method for merging multiple parameter sources into a
      unified configuration instance.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Type, TypeVar

import matplotlib.pyplot as plt

from frequenz.lib.notebooks.solar.maintenance.translator import TranslationManager

T = TypeVar("T", bound="BaseViewConfig")


@dataclass
class BaseViewConfig:
    """Base configuration for plot data preparation and visualisation."""

    translation_manager: TranslationManager = field(
        default_factory=TranslationManager,
        metadata={
            "description": "Instance of TranslationManager to handle translations",
        },
    )

    x_axis_label: str = field(
        default="x-axis",
        metadata={
            "description": "Data label for the x-axis",
        },
    )

    string_separator: str = field(
        default="\n",
        metadata={
            "description": "Separator for string concatenation",
        },
    )

    verbose: bool = field(  # temporary field; will be replaced by a logger
        default=False,
        metadata={
            "description": "Verbosity flag for logging and debugging",
        },
    )

    @classmethod
    def create_config(cls: Type[T], *extra_params: dict[str, Any], **kwargs: Any) -> T:
        """Create a plot configuration instance by merging multiple parameter sources.

        This method allows merging multiple configuration dictionaries
        (`*extra_params`) and keyword arguments (`**kwargs`). Later dictionaries
        and keyword arguments override earlier ones. It is useful for combining
        base configurations, user overrides, and dynamic parameters.

        Args:
            *extra_params: One or more dictionaries to merge as configuration.
            **kwargs: Additional keyword arguments that override earlier parameters.

        Returns:
            An instance of the plot configuration class with the merged parameters.
        """
        valid_fields = {f.name for f in fields(cls)}
        combined_params = {k: v for d in extra_params for k, v in d.items()}
        combined_params.update(kwargs)
        valid_params = {k: v for k, v in combined_params.items() if k in valid_fields}
        return cls(**valid_params)


@dataclass
class CalendarViewConfig(BaseViewConfig):
    """Configuration for calendar view plot data."""

    time_frame: str = field(
        default="day",
        metadata={
            "description": "Time frame for calendar view",
            "validate": lambda x: x in ["day", "month", "year", "12month", "all"],
        },
    )


@dataclass
class RollingViewConfig(BaseViewConfig):
    """Configuration for rolling view plot data."""

    view: tuple[int, str] = field(
        default=(30, "days"),
        metadata={
            "description": "Duration and time frame for the rolling view",
        },
    )

    rolling_average: bool = field(
        default=False,
        metadata={
            "description": "Whether to compute a rolling average",
        },
    )

    primary_colour: str = field(
        default=plt.rcParams["lines.color"],
        metadata={
            "description": "Primary colour to use for the plot",
        },
    )

    cmap_name: str = field(
        default=plt.rcParams["image.cmap"],
        metadata={
            "description": "Name of the colormap to use",
        },
    )


@dataclass
class ProfileViewConfig(BaseViewConfig):
    """Configuration for statistical production profile plot data."""

    groupings: list[str] = field(
        default_factory=list,
        metadata={
            "description": ("List of groupings for statistical analysis "),
            "validate": lambda x: x in ["grouped", "continuous", "24h_continuous"],
        },
    )

    duration: int = field(
        default=30,
        metadata={
            "description": "Duration in days for statistical analysis",
        },
    )

    column_label: str = field(
        default="energy_kWh",
        metadata={
            "description": "Column label for the energy variable to plot",
        },
    )

    cmap_name: str = field(
        default=plt.rcParams["image.cmap"],
        metadata={
            "description": "Name of the colormap to use",
        },
    )

    interpolate_colormap: bool = field(
        default=False,
        metadata={
            "description": "Whether to interpolate the colormap",
        },
    )


@dataclass
class DailyViewConfig(BaseViewConfig):
    """Configuration for daily production plot data."""

    column_label: str = field(
        default="energy_kWh",
        metadata={
            "description": "Column label to use for daily production plots",
        },
    )

    colour: str = field(
        default=plt.rcParams["lines.color"],
        metadata={
            "description": "Colour to use for the daily production plot",
        },
    )


@dataclass
class StatsViewConfig(BaseViewConfig):
    """Configuration for production statistics view."""

    # No additional fields; inherits all from BaseViewConfig.
