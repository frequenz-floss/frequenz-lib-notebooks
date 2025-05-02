# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Module for plot style strategies for the plotter module.

This module contains the PlotStyleStrategy abstract base class and its subclasses
for different types of solar production profiles. The subclasses implement the
`get_plot_styles` method to return the plot styles based on the time frame and
additional parameters. It also contains the style_table function to convert a
DataFrame to a styled HTML table (currently fixed to the Frequenz-Neustrom brand
colours).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, display

from frequenz.lib.notebooks.solar.maintenance.translator import TranslationManager


class PlotStyleStrategy(ABC):
    """Abstract base class for plot style strategies.

    Subclasses must implement the `get_plot_styles` method based on the time
    frame and additional keyword arguments. The exact required parameters in
    `kwargs` will vary depending on the specific plot style.
    """

    @abstractmethod
    def get_plot_styles(
        self,
        time_frame: str,
        translation_manager: TranslationManager = TranslationManager(),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return the plot styles based on the time frame and other parameters.

        Args:
            time_frame: The time frame for the plot. The valid options will
                depend on the specific subclass implementation.
            translation_manager: Instance of TranslationManager to handle
                translations.
            **kwargs: Additional parameters required or optional by the specific
                subclass implementation.

        Returns:
            A dictionary containing plot styles relevant to the subclass.
        """

    @classmethod
    def get_strategy(cls, plot_type: str) -> "PlotStyleStrategy":
        """Return the appropriate strategy class based on plot type.

        Args:
            plot_type: The type of plot style strategy to use.

        Returns:
            The appropriate subclass of PlotStyleStrategy.

        Raises:
            ValueError: If the plot type is unsupported.
        """
        strategies: dict[str, type[PlotStyleStrategy]] = {
            "statistical": StatisticalPlotStyle,
            "rolling": RollingPlotStyle,
            "calendar": CalendarPlotStyle,
        }

        strategy = strategies.get(plot_type, None)
        if strategy is None:
            raise ValueError(
                f"Unsupported plot type: {plot_type}. "
                f"Supported types: {', '.join(strategies.keys())}"
            )
        return strategy()

    def _validate_required_params(
        self, params: list[str], kwargs: dict[str, Any]
    ) -> None:
        """Validate that required parameters are present in kwargs.

        Args:
            params: List of required parameter names.
            kwargs: Dictionary of keyword arguments.

        Raises:
            ValueError: If any required parameters are missing.
        """
        missing_params = [param for param in params if kwargs.get(param) is None]
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

    def _validate_time_frame(
        self, time_frame: str, valid_time_frames: list[str]
    ) -> None:
        """Validate the given time_frame against valid options.

        Args:
            time_frame: The time frame to validate.
            valid_time_frames: A list of valid time frame values.

        Raises:
            ValueError: If time_frame is not in valid_time_frames.
        """
        if time_frame not in valid_time_frames:
            raise ValueError(
                f"Invalid time_frame: {time_frame}. "
                f"Valid options are: {', '.join(valid_time_frames)}"
            )


class StatisticalPlotStyle(PlotStyleStrategy):
    """Plot style strategy for statistical production profile."""

    def get_plot_styles(
        self,
        time_frame: str,
        translation_manager: TranslationManager = TranslationManager(),
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        """Return the plot styles for statistical profiles based on the time frame.

        Args:
            time_frame: The time frame for the plot. Valid options are 'grouped',
                'continuous', '24h_continuous'.
            translation_manager: Instance of TranslationManager to handle
                translations.
            **kwargs: The following additional parameters:
                - data_index (pd.DatetimeIndex): (Required) The index of the data.
                - col_label (str): (Optional) Column label for the data being
                    plotted.
                - cmap (Colormap): (Optional) The colormap to use for the plot.

        Returns:
            A dictionary containing axes parameters and plot styles for statistics.
        """
        self._validate_required_params(["data_index"], kwargs)
        self._validate_time_frame(
            time_frame, ["grouped", "continuous", "24h_continuous"]
        )

        data_index = kwargs.pop("data_index")
        col_label = kwargs.pop("col_label", "")
        cmap = kwargs.pop(
            "cmap", plt.get_cmap(plt.rcParams.get("image.cmap", "viridis"))
        )
        interpolate_colormap = kwargs.pop("interpolate_colormap", False)

        y_label = " (".join(col_label.split("_")) + ")" if col_label else ""
        try:
            duration = int(data_index.to_series().diff().mode()[0].total_seconds() / 60)
        except TypeError:
            duration = int(
                data_index.map(lambda t: datetime.combine(datetime.today(), t))
                .to_series()
                .diff()
                .mode()[0]
                .total_seconds()
                / 60
            )
        except KeyError:
            duration = None  # occurs when data_index is empty or has a single value

        axes_params: dict[str, dict[str, str | int | list[str] | None]] = {
            "grouped": {
                "y_label": translation_manager.translate(y_label.capitalize()),
                "x_label": translation_manager.translate("Time of day"),
                "title": translation_manager.translate(
                    "{value}-min profile", value=duration
                ),
                "max_xticks": 10,
            },
            "continuous": {
                "y_label": translation_manager.translate(y_label.capitalize()),
                "x_label": translation_manager.translate("Day of month / Hour of day"),
                "title": translation_manager.translate(
                    "Daily profile ({value}-min intervals)", value=duration
                ),
                "max_xticks": 15,
            },
            "24h_continuous": {
                "y_label": translation_manager.translate(y_label.capitalize()),
                "x_label": translation_manager.translate("Day of month"),
                "title": translation_manager.translate(
                    "Daily profile ({value}-hour intervals)", value=24
                ),
                "max_xticks": 15,
            },
        }
        statistics_plot_styles: dict[
            str, dict[str, str | float | tuple[float, ...] | None]
        ] = {
            translation_manager.translate("mean"): {
                "marker": "o",
                "color": cmap(0.8) if interpolate_colormap else cmap(2),
                "kind": "line",
                "alpha": 0.8,
            },
            translation_manager.translate("median"): {
                "marker": "s",
                "color": cmap(0.4) if interpolate_colormap else cmap(5),
                "kind": "line",
                "alpha": 0.8,
            },
            "min": {
                "marker": "v",
                "color": cmap(0.2) if interpolate_colormap else cmap(11),
                "kind": "area",
                "curve_2": "max",
                "alpha": 0.4,
                "area_label": "Min-Max",
            },
            "max": {"marker": "^", "color": cmap(0.9), "kind": None},
            "25th percentile": {
                "marker": "P",
                "color": cmap(0.5) if interpolate_colormap else cmap(4),
                "kind": "area",
                "curve_2": "75th percentile",
                "alpha": 0.4,
                "area_label": "Q1-Q3",
            },
            "75th percentile": {"marker": "D", "color": cmap(0.7), "kind": None},
        }
        return {
            "axes_params": axes_params[time_frame],
            "statistics_plot_styles": statistics_plot_styles,
        }


class RollingPlotStyle(PlotStyleStrategy):
    """Plot style strategy for rolling production profile."""

    def get_plot_styles(
        self,
        time_frame: str,
        translation_manager: TranslationManager = TranslationManager(),
        **kwargs: Any,
    ) -> dict[str, str | float]:
        """Return the plot styles for rolling profiles based on the time frame.

        Args:
            time_frame: The time frame for the plot. Valid options are 'hours',
                'days'.
            translation_manager: Instance of TranslationManager to handle
                translations.
            **kwargs: The following additional parameters:
                - timezone (str): (Required) Timezone for the plot.
                - current (datetime): (Required) Current time reference for the
                    plot.
                - time (float): (Required) The time span for the rolling window.
                - timespan (timedelta): (Required) Total timespan for data
                    aggregation.
                - rolling_average (bool): (Required) Flag to indicate if a
                    rolling average is used.
                - colour (str): (Optional) Colour for the plot.

        Returns:
            A dictionary containing plot styles for rolling averages.
        """
        self._validate_required_params(
            ["timezone", "current", "time", "rolling_average"], kwargs
        )
        self._validate_time_frame(time_frame, ["hours", "days"])

        timezone = kwargs.pop("timezone")
        current = kwargs.pop("current")
        time = kwargs.pop("time")
        rolling_average = kwargs.pop("rolling_average")
        colour = kwargs.pop("colour", plt.rcParams["lines.color"])

        plot_styles: dict[str, dict[str, str | float]] = {
            "hours": {
                "plot_style": "--o",
                "plot_kind": "line",
                "x_label": translation_manager.translate(
                    "Time ({value})", value=translation_manager.translate(timezone)
                ),
                "title": (
                    translation_manager.translate(
                        "Production in the past {value_1}h (reference time: {value_2})",
                        value_1=time,
                        value_2=current.strftime("%d %b, %Y %H:%M:%S"),
                    )
                    if not rolling_average
                    else translation_manager.translate(
                        "Rolling {value}-hour Average Production", value=time
                    )
                ),
                "color": colour,
                "max_xticks": 10,
                "group_size_rolling_average_plot": 24 * 365,
            },
            "days": {
                "plot_style": "--o",
                "plot_kind": "line",
                "x_label": translation_manager.translate("Month/Day"),
                "title": (
                    translation_manager.translate(
                        "Production in the past {value} days", value=time
                    )
                    if not rolling_average
                    else translation_manager.translate(
                        "Rolling {value}-day Average Production", value=time
                    )
                ),
                "color": colour,
                "alpha": 0.7,
                "max_xticks": 15,
                "group_size_rolling_average_plot": 365,
            },
        }
        return plot_styles[time_frame]


class CalendarPlotStyle(PlotStyleStrategy):
    """Plot style strategy for calendar production profile."""

    def get_plot_styles(
        self,
        time_frame: str,
        translation_manager: TranslationManager = TranslationManager(),
        **kwargs: Any,
    ) -> dict[str, str]:
        """Return the plot styles for calendar profiles based on the time frame.

        Args:
            time_frame: The time frame for the calendar view. Valid options are
                'day', 'month', 'year', '12month', 'all'.
            translation_manager: Instance of TranslationManager to handle
                translations.
            **kwargs: The following additional parameters:
                - timezone (str): (Required) Timezone for the plot.
                - current (datetime): (Required) Current time reference for the
                    plot.

        Returns:
            A dictionary containing plot styles for the calendar view.
        """
        self._validate_required_params(["timezone", "current"], kwargs)
        self._validate_time_frame(
            time_frame, ["day", "month", "year", "12month", "all"]
        )

        timezone = kwargs.pop("timezone")
        current = kwargs.pop("current")

        plot_styles: dict[str, dict[str, str]] = {
            "day": {
                "plot_style": "--o",
                "plot_var": "power_kW",
                "y_label": translation_manager.translate("Power (kW)"),
                "x_label": translation_manager.translate(
                    "Time ({value})", value=translation_manager.translate(timezone)
                ),
                "title": translation_manager.translate(
                    "Production on {value}", value=current.date()
                ),
            },
            "month": {
                "plot_style": "",
                "plot_var": "energy_kWh",
                "y_label": translation_manager.translate("Energy (kWh)"),
                "x_label": translation_manager.translate("Day of month"),
                "title": translation_manager.translate(
                    "Production in {value}", value=current.month_name()
                ),
            },
            "year": {
                "plot_style": "",
                "plot_var": "energy_MWh",
                "y_label": translation_manager.translate("Energy (MWh)"),
                "x_label": translation_manager.translate("Month in year"),
                "title": translation_manager.translate(
                    "Production in {value}", value=current.year
                ),
            },
            "12month": {
                "plot_style": "",
                "plot_var": "energy_MWh",
                "y_label": translation_manager.translate("Energy (MWh)"),
                "x_label": translation_manager.translate("Year/Month"),
                "title": translation_manager.translate(
                    "Production in the past {value} months", value=12
                ),
            },
            "all": {
                "plot_style": "",
                "plot_var": "energy_MWh",
                "y_label": translation_manager.translate("Energy (MWh)"),
                "x_label": translation_manager.translate("Year"),
                "title": translation_manager.translate("Yearly production"),
            },
        }
        return plot_styles[time_frame]


def style_table(df: pd.DataFrame, show: bool = False) -> HTML:
    """Convert the DataFrame to a styled HTML table.

    Args:
        df: The DataFrame to convert to an HTML table.
        show: A flag to display the HTML table.

    Returns:
        An HTML object containing the styled table.
    """
    html_table = df.to_html(border=0, index=True)
    styled_html = _generate_table_style_freqstrom() + html_table
    if show:
        display(HTML(styled_html))  # type: ignore
    return HTML(styled_html)  # type: ignore


def _generate_table_style_freqstrom() -> str:
    """Return the HTML string with the Frequenz-Neustrom styles.

    Note: The colours are based on the Frequenz-Neustrom brand colours.

    Returns:
        The HTML string containing the custom table styles.
    """
    return """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Roboto', 'HelveticaNeue', 'Arial', sans-serif;
            margin: 20px 0;
            font-size: 14px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        thead th {
            background-color: #00FFCE;
            color: #1A1A1A;
            text-align: center;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
        }
        tbody td {
            text-align: center;
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tbody tr:nth-child(odd) {
            background-color: #F4F4F4;
        }
        tbody tr:nth-child(even) {
            background-color: #D5D5D5;
        }
        tbody tr:hover {
            background-color: #CEFFF6;
            color: #1A1A1A;
        }
        tfoot td {
            background-color: #979797;
            color: #333;
            font-size: 14px;
            text-align: center;
            padding: 12px;
        }
    </style>
    """
