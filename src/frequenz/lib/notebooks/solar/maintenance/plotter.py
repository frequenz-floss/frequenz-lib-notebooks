# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
Module for plotting data.

This module provides classes for creating various types of plots. Each plotter
class is designed to work with a specific data preparer class from the
`plotter_data_preparer` module and uses configurations defined in the
`plotter_config` module.

The plotter classes rely on the prepared data and configuration objects to create
visualisations. They encapsulate the logic for handling figure and axis
initialisation, plot styling, x-tick generation, legends, and other plotting
features.

Classes:
- `BasePlotter`: Abstract base class for all plotters.
- `CalendarPlotter`: Generates calendar view plots for solar power production.
- `RollingPlotter`: Generates rolling view plots with optional rolling averages.
- `ProfilePlotter`: Generates statistical profile plots based on data groupings.
- `DailyPlotter`: Generates daily production plots.

Functions:
- `_get_plot_styles`: Retrieves plot styling details for the specified plot type
    and time frame.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from frequenz.lib.notebooks.solar.maintenance.plot_styles import PlotStyleStrategy
from frequenz.lib.notebooks.solar.maintenance.plotter_config import (
    CalendarViewConfig,
    DailyViewConfig,
    ProfileViewConfig,
    RollingViewConfig,
)
from frequenz.lib.notebooks.solar.maintenance.translator import TranslationManager


class BasePlotter(ABC):
    """Abstract base class for plotters."""

    def __init__(self, config: Any):
        """Initialise the plotter with the configuration.

        Args:
            config: The plotting configuration.
        """
        self.config = config
        self.fig_size = plt.rcParams["figure.figsize"]
        self._production_artist_zorder = 2.02  # zorder for production artists

    @abstractmethod
    def plot(
        self, data: pd.DataFrame, fig: Figure | None = None, ax: Axes | None = None
    ) -> None:
        """Plot the data.

        Args:
            data: Data to be plotted, prepared by the corresponding preparer.
            fig: The matplotlib figure for the plot.
            ax: The matplotlib axis for the plot.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement the plot method.")

    def _is_data_empty(self, df: pd.DataFrame, text: str, verbose: bool) -> bool:
        """Check if the DataFrame is empty.

        Args:
            df: DataFrame to check for emptiness.
            text: Text to display if the DataFrame is empty.
            verbose: A boolean flag to display the text if the DataFrame is empty.

        Returns:
            A boolean flag indicating if the DataFrame is empty.
        """
        if df.empty:
            if verbose:
                print(f"No data to plot. {text}")
            return True
        return False

    def _initialise_figure(
        self, fig: Figure | None = None, ax: Axes | None = None
    ) -> tuple[Figure, Axes]:
        """Initialise the plot figure and axis.

        Args:
            fig: The matplotlib figure to plot the data on. If not provided, a new plot
                is created. Note that both fig and ax must be provided together.
            ax: The matplotlib axis to plot the data. If not provided, a new plot
                is created. Note that both fig and ax must be provided together.

        Returns:
            A tuple containing the figure and axis objects.

        Raises:
            ValueError: If either fig or ax are provided without
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size)
        elif fig is None or ax is None:
            raise ValueError("Either define both figure and axes or neither.")
        return fig, ax

    @staticmethod
    def _hide_axes(ax: Axes | None) -> None:
        """Hide the axes in the plot.

        Args:
            ax: The matplotlib axis to hide.
        """
        if ax is not None:
            ax.set_visible(False)

    def _generate_xticks(
        self,
        *,
        arr: list[str],
        max_xticks: int = 15,
        sep: str | None = None,
        indices: list[int] | None = None,
    ) -> tuple[list[str], list[int]]:
        """Generate x-axis labels and ticks based on the input array and max_xticks.

        Args:
            arr: The list containing the strings to generate the x-axis labels and
                ticks from.
            max_xticks: The maximum number of x-ticks to display.
            sep: The separator used to split the x-axis labels. If provided, the
                function will split each label and only display the unique values
                after the separator.
            indices: The list of indices to use for the x-axis ticks. If provided,
                the function will use these indices to generate the x-axis ticks.

        Returns:
            A tuple containing the x-axis tick labels and tick positions.
        """
        if indices:
            arr = [arr[idx] for idx in indices]
        _indices = self._find_first_occurrences(arr)
        if (length := len(_indices)) > max_xticks:
            step = max(1, length // max_xticks + length % 2)
            xticklabels = [str(arr[idx]) for idx in _indices[::step]]
            xticks = (
                [indices[idx] for idx in _indices][::step]
                if indices
                else _indices[::step]
            )
        else:
            xticklabels = [str(str(arr[idx])) for idx in _indices]
            xticks = [indices[idx] for idx in _indices] if indices else _indices
        if sep:
            xticklabels = self._format_labels(xticklabels, sep)
        return xticklabels, xticks

    @staticmethod
    def _find_first_occurrences(arr: list[str]) -> list[int]:
        """Find the first occurrences of unique elements in the input list.

        Args:
            arr: The list of strings to find the first occurrences of.

        Returns:
            A list containing the indices of the first occurrences of unique
            elements in the input list.
        """
        seen = set()
        first_occurrences = []
        for idx, item in enumerate(arr):
            if item not in seen:
                seen.add(item)
                first_occurrences.append(idx)
        return first_occurrences

    @staticmethod
    def _format_labels(labels: list[str], sep: str) -> list[str]:
        """Format labels to show only unique values for each date component.

        Expected label format: "day{sep}month{sep}year{sep}time".

        Args:
            labels: List of input labels to format.
            sep: Separator used to split parts of the labels.

        Returns:
            A list of formatted labels.
        """
        previous_parts: list[str] = ["", "", "", ""]
        formatted_labels = []
        for label in labels:
            parts = label.split(sep)
            day, month, year, *time_list = parts + [""] * (4 - len(parts))
            time = time_list[0]

            # Determine what parts to show based on changes from previous label
            if year != previous_parts[2]:
                formatted_labels.append(label)
            elif month != previous_parts[1]:
                formatted_labels.append(
                    f"{day}{sep}{month}"
                    + (f"{sep}{time}" if time and time != previous_parts[3] else "")
                )
            elif day != previous_parts[0]:
                formatted_labels.append(
                    f"{day}"
                    + (f"{sep}{time}" if time and time != previous_parts[3] else "")
                )
            elif time != previous_parts[3]:
                formatted_labels.append(time)
            else:
                formatted_labels.append(day)
            previous_parts = [day, month, year, time]
        return formatted_labels

    @staticmethod
    def _set_axes_properties(ax: Axes, properties: dict[str, Any]) -> None:
        """Set common axes properties such as title, labels, and limits.

        Args:
            ax: The matplotlib axis to apply the styles to.
            properties: A dictionary containing keys for title, x_label, y_label,
                and other styling options supported by the matplotlib Axes.
        """
        property_mapping: dict[str, Callable[[Any], Any] | None] = {
            "title": ax.set_title,
            "x_label": ax.set_xlabel,
            "y_label": ax.set_ylabel,
            "legend_visible": lambda val: ax.legend().set_visible(val),
            "xlim": ax.set_xlim,
            "ylim": ax.set_ylim,
            "xticks": ax.set_xticks,
            "xticklabels": ax.set_xticklabels,
        }
        for prop, value in properties.items():
            apply_func = property_mapping.get(prop, None)
            if isinstance(apply_func, type(None)):
                warn(f"Property {prop} not found in supported properties.")
            else:
                if value is not None:
                    apply_func(value)

    @staticmethod
    def _add_figure_legend(
        fig: Figure,
        axs: list[Axes],
        loc: str = "upper right",
        ncol: int = 1,
        bbox_to_anchor: tuple[float, float] | None = None,
    ) -> None:
        """Add a legend to the figure based on the visible axes.

        This function collects handles and labels from all visible axes, removes
        duplicates, and adds a legend to the figure. It allows for flexible
        positioning of the legend using `bbox_to_anchor` and `loc`.

        Args:
            fig: The matplotlib figure to which the legend should be added.
            axs: A list of matplotlib axes or a single axis to extract handles/labels from.
            loc: The location of the legend on the figure.
            ncol: The number of columns for the legend.
            bbox_to_anchor: The position of the legend on the figure. (0, 0) is the
                lower-left corner, and (1, 1) is the upper-right corner of the figure.
                This is useful for placing the legend outside the axes or figure.
        """
        handles, labels = [], []
        seen = set()
        for ax in axs:
            if ax.get_visible():
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label not in seen:
                        seen.add(label)
                        handles.append(handle)
                        labels.append(label)
        if handles and labels:
            fig.legend(
                handles,
                labels,
                bbox_to_anchor=bbox_to_anchor,
                loc=loc,
                ncol=ncol,
            )
            fig.tight_layout()
        else:
            # NOTE: log message: There is no data to plot, closing the figure.
            plt.close(fig)


class CalendarPlotter(BasePlotter):
    """Plotter for calendar view plots."""

    def __init__(self, config: CalendarViewConfig) -> None:
        """Initialise the calendar plotter with the configuration.

        Args:
            config: The plotting configuration for calendar view plots.
        """
        super().__init__(config)

    def plot(
        self, data: pd.DataFrame, fig: Figure | None = None, ax: Axes | None = None
    ) -> None:
        """Plot the calendar view.

        Args:
            data: Prepared data for the calendar view plot.
            fig: The matplotlib figure to plot the data on. Note that both fig
                and ax must be provided together.
            ax: The matplotlib axis to plot the data. If not provided, a new plot
                is created. Note that both fig and ax must be provided together.
        """
        if self._is_data_empty(data, "(CalendarPlotter)", self.config.verbose):
            self._hide_axes(ax)
            return
        fig, ax = self._initialise_figure(fig=fig, ax=ax)
        self._plot(data, ax)

    def _plot(self, data: pd.DataFrame, ax: Axes) -> None:
        """Plot the calendar data.

        Args:
            data: Data for the calendar plot.
            ax: The matplotlib axis for the plot.

        Raises:
            ValueError: If an invalid time frame is provided.
        """
        _data_index = pd.DatetimeIndex(data.index)
        current = _data_index[-1]
        time_frame_mapping: dict[str, dict[str, Any]] = {
            "day": {
                "xlim": (
                    current.replace(hour=0, minute=0, second=0),
                    current.replace(hour=23, minute=59, second=59),
                ),
                "xticklabels": None,
            },
            "month": {
                "xlim": None,
                "xticklabels": _data_index.day.unique()
                .sort_values()
                .values.astype(str),
            },
            "year": {
                "xlim": None,
                "xticklabels": _data_index.month.unique()
                .sort_values()
                .values.astype(str),
            },
            "12month": {
                "xlim": None,
                "xticklabels": [
                    "/".join(subitem[-2:] for subitem in str(item).split("-")[:2])
                    for item in _data_index
                ],
            },
            "all": {"xlim": None, "xticklabels": _data_index.year.unique()},
        }
        try:
            xlim, xticklabels = time_frame_mapping[self.config.time_frame].values()
        except KeyError as exc:
            raise ValueError(f"Invalid time frame: {self.config.time_frame}") from exc

        plot_styles = _get_plot_styles(
            plot_type="calendar",
            time_frame=self.config.time_frame.lower(),
            translation_manager=self.config.translation_manager,
            timezone=str(_data_index.tzinfo),
            current=current,
        )
        data.plot(
            ax=ax,
            kind="line" if self.config.time_frame.lower() == "day" else "bar",
            style=plot_styles["plot_style"],
            y=plot_styles["plot_var"],
            rot=0,
            zorder=self._production_artist_zorder,
            legend=False,
        )
        self._set_axes_properties(
            ax=ax,
            properties={
                "title": plot_styles["title"],
                "x_label": plot_styles["x_label"],
                "y_label": plot_styles["y_label"],
                "xlim": xlim if xlim is not None else None,
                "xticklabels": (
                    self.config.translation_manager.translate_list(
                        xticklabels, format_numbers=False
                    )
                    if xticklabels
                    else None
                ),
                "hide_legend": False,
            },
        )


class RollingPlotter(BasePlotter):
    """Plotter for rolling view plots."""

    def __init__(self, config: RollingViewConfig) -> None:
        """Initialise the rolling plotter with the configuration.

        Args:
            config: The plotting configuration for rolling view plots.
        """
        super().__init__(config)

    def plot(
        self, data: pd.DataFrame, fig: Figure | None = None, ax: Axes | None = None
    ) -> None:
        """Plot the rolling view.

        Args:
            data: Prepared data for the rolling view plot.
            fig: The matplotlib figure to plot the data on. Note that both fig
                and ax must be provided together.
            ax: The matplotlib axis to plot the data. If not provided, a new plot
                is created. Note that both fig and ax must be provided together.
        """
        if self._is_data_empty(data, "(RollingPlotter)", self.config.verbose):
            self._hide_axes(ax)
            return
        fig, ax = self._initialise_figure(fig=fig, ax=ax)
        self._plot(data, fig, ax)

    # pylint: disable=too-many-locals
    def _plot(self, data: pd.DataFrame, fig: Figure, ax: Axes) -> None:
        """Plot the rolling data.

        Args:
            data: Data for the rolling plot.
            fig: The matplotlib figure for the plot.
            ax: The matplotlib axis for the plot.
        """
        data_index = pd.to_datetime(data.index)
        cmap = plt.get_cmap(self.config.cmap_name)
        plot_styles = _get_plot_styles(
            plot_type="rolling",
            time_frame=self.config.view[1],
            translation_manager=self.config.translation_manager,
            timezone=str(data_index.tzinfo),
            current=data_index[-1],
            time=self.config.view[0],
            rolling_average=self.config.rolling_average,
            colour=self.config.primary_colour,
        )

        if self.config.rolling_average:
            ngroups = 1 + (
                data.shape[0] // plot_styles["group_size_rolling_average_plot"]
            )
            cmap_values = np.linspace(0, 0.9, ngroups)
            for i in range(0, ngroups, 1):
                oldest_index = max(
                    -plot_styles["group_size_rolling_average_plot"] * (i + 1),
                    -data.shape[0],
                )
                newest_index = (
                    -plot_styles["group_size_rolling_average_plot"] * i
                    if i > 0
                    else None
                )
                if i == 0:
                    label = self.config.translation_manager.translate(
                        "current {value}-" + f"{self.config.view[1][:-1]} cycle",
                        value=plot_styles["group_size_rolling_average_plot"],
                    )
                elif i == 1:
                    label = self.config.translation_manager.translate("1 cycle ago")
                else:
                    label = self.config.translation_manager.translate(
                        "{value} cycles ago", value=i
                    )
                ax.plot(
                    data[self.config.x_axis_label].values[oldest_index:newest_index],
                    data[
                        [col for col in data.columns if col != self.config.x_axis_label]
                    ].values[oldest_index:newest_index],
                    label=label,
                    color=cmap(cmap_values[i]),
                    alpha=plot_styles["alpha"],
                )
            xticklabels, xticks = self._generate_xticks(
                arr=list(
                    data.iloc[-plot_styles["group_size_rolling_average_plot"] :][
                        self.config.x_axis_label
                    ]
                ),
                max_xticks=plot_styles["max_xticks"],
                sep=self.config.string_separator,
            )
            self._add_figure_legend(
                fig, [ax], loc="center left", ncol=1, bbox_to_anchor=(1.0, 0.5)
            )

        else:
            xticklabels, xticks = self._generate_xticks(
                arr=list(data[self.config.x_axis_label].values),
                max_xticks=plot_styles["max_xticks"],
                sep=(
                    self.config.string_separator
                    if self.config.view[1] == "days"
                    else None
                ),
            )
            cols_to_plot = [
                col for col in data.columns if col != self.config.x_axis_label
            ]
            cmap_values = np.linspace(0, 0.9, len(cols_to_plot))
            data.plot(
                ax=ax,
                kind=plot_styles["plot_kind"],
                style=str(plot_styles["plot_style"]),
                x=self.config.x_axis_label,
                y=cols_to_plot,
                label=self.config.translation_manager.translate_list(cols_to_plot),
                color=(
                    plot_styles["color"]
                    if len(cols_to_plot) == 1
                    else [cmap(cmap_values[i]) for i, _ in enumerate(cols_to_plot)]
                ),
                rot=0,
                zorder=self._production_artist_zorder,
            )
        self._set_axes_properties(
            ax=ax,
            properties={
                "title": str(plot_styles["title"]),
                "x_label": str(plot_styles["x_label"]),
                "xticks": xticks,
                "xticklabels": self.config.translation_manager.translate_list(
                    xticklabels, format_numbers=False
                ),
            },
        )


class ProfilePlotter(BasePlotter):
    """Plotter for statistifcal profile plots."""

    def __init__(self, config: ProfileViewConfig) -> None:
        """Initialise the profile plotter with the configuration.

        Args:
            config: The plotting configuration for statistical profile plots.
        """
        super().__init__(config)
        # NOTE: The variable below should be removed in the future as it
        # hardcodes a group label that is defined in config.py and might change
        # in the future which would make this hardcoding invalid.
        self._group_label_constant = "grouped"

    def plot(
        self,
        data: pd.DataFrame,
        fig: Figure | None = None,
        ax: Axes | None = None,
        group_label: str = "",
    ) -> None:
        """Plot the statistical profile.

        Args:
            data: Prepared data for the statistical profile plot.
            fig: The matplotlib figure to plot the data on. Note that both fig
                and ax must be provided together.
            ax: The matplotlib axis to plot the data. If not provided, a new plot
                is created. Note that both fig and ax must be provided together.
            group_label: The grouping label used for the statistical analysis
                that generated the data to be plotted.
        """
        if self._is_data_empty(data, "(statistical profile view)", self.config.verbose):
            self._hide_axes(ax)
            return
        fig, ax = self._initialise_figure(fig=fig, ax=ax)
        self._plot(data, group_label, fig, ax)

    # pylint: disable-next=too-many-locals
    def _plot(
        self, data: pd.DataFrame, group_label: str, fig: Figure, ax: Axes
    ) -> None:
        """Plot the statistical profile.

        Args:
            data: Data for the statistical profile plot.
            group_label: The grouping label used for the statistical analysis
                that generated the data to be plotted.
            fig: The matplotlib figure for the plot.
            ax: The matplotlib axis for the plot.
        """
        cols_to_plot = [col for col in data.columns if self.config.column_label in col]
        labels = [col.split("_")[-1] for col in cols_to_plot]
        plot_styles = _get_plot_styles(
            plot_type="statistical",
            time_frame=group_label,
            translation_manager=self.config.translation_manager,
            col_label=self.config.column_label,
            cmap=plt.get_cmap(self.config.cmap_name),
            interpolate_colormap=self.config.interpolate_colormap,
            data_index=data.index,
        )

        if len(data) == 1:
            start = data.index[0] - pd.Timedelta(hours=1)
            end = data.index[0] + pd.Timedelta(hours=1)
            for _label, _col in zip(labels, cols_to_plot):
                _label = self.config.translation_manager.translate(_label)
                data.reset_index().plot(
                    ax=ax,
                    kind="scatter",
                    x=data.index.name,
                    y=_col,
                    marker=plot_styles["statistics_plot_styles"][_label]["marker"],
                    label=_label,
                    color=plot_styles["statistics_plot_styles"][_label]["color"],
                    xlim=(start, end),
                    zorder=self._production_artist_zorder,
                )
            xticks = None
            xticklabels = None
        else:
            for _label, _col in zip(labels, cols_to_plot):
                _label = self.config.translation_manager.translate(_label)
                if plot_styles["statistics_plot_styles"][_label]["kind"] == "line":
                    ax.plot(
                        data[self.config.x_axis_label].values.astype("object"),
                        data[_col].values.astype("float"),
                        color=plot_styles["statistics_plot_styles"][_label]["color"],
                        alpha=plot_styles["statistics_plot_styles"][_label]["alpha"],
                        label=_label,
                        zorder=self._production_artist_zorder,
                    )
                elif plot_styles["statistics_plot_styles"][_label]["kind"] == "area":
                    ax.fill_between(
                        data[self.config.x_axis_label].values.astype("object"),
                        data[_col].values.astype("float"),
                        data[
                            f"{self.config.column_label}_"
                            f"{plot_styles['statistics_plot_styles'][_label]['curve_2']}"
                        ].values.astype("float"),
                        color=plot_styles["statistics_plot_styles"][_label]["color"],
                        alpha=plot_styles["statistics_plot_styles"][_label]["alpha"],
                        label=plot_styles["statistics_plot_styles"][_label][
                            "area_label"
                        ],
                    )
                else:
                    continue

            # special case for "continuous" plot: ensure xticks are at the beginning of each day
            indices_mask: list[int] | None = None
            if group_label == "continuous":
                df_index = pd.to_datetime(data.index)
                indices_mask = np.where(~df_index.normalize().duplicated(keep="first"))[
                    0
                ].tolist()
            xticklabels, xticks = self._generate_xticks(
                arr=list(data[self.config.x_axis_label].values),
                max_xticks=int(plot_styles["axes_params"]["max_xticks"]),
                sep=(
                    None
                    if group_label in [self._group_label_constant]
                    else self.config.string_separator
                ),
                indices=indices_mask,
            )
        self._set_axes_properties(
            ax=ax,
            properties={
                "title": str(plot_styles["axes_params"]["title"]),
                "x_label": str(plot_styles["axes_params"]["x_label"]),
                "y_label": str(plot_styles["axes_params"]["y_label"]),
                "legend_visible": False,
                "xticks": xticks if xticks is not None else None,
                "xticklabels": (
                    self.config.translation_manager.translate_list(
                        xticklabels, format_numbers=False
                    )
                    if xticklabels is not None
                    else None
                ),
            },
        )
        self._add_figure_legend(
            fig,
            [ax],
            loc="lower center",
            ncol=len(labels) // 2,
            bbox_to_anchor=(0.5, -0.03),
        )


class DailyPlotter(BasePlotter):
    """Plotter for daily view plots."""

    def __init__(self, config: DailyViewConfig) -> None:
        """Initialise the daily plotter with the configuration.

        Args:
            config: The plotting configuration for daily view plots.
        """
        super().__init__(config)

    def plot(
        self, data: pd.DataFrame, fig: Figure | None = None, ax: Axes | None = None
    ) -> None:
        """Plot the daily view.

        Args:
            data: Prepared data for the daily view plot.
            fig: The matplotlib figure to plot the data on. Note that both fig
                and ax must be provided together.
            ax: The matplotlib axis to plot the data. If not provided, a new plot
                is created. Note that both fig and ax must be provided together.
        """
        if self._is_data_empty(data, "(daily view)", self.config.verbose):
            self._hide_axes(ax)
            return
        fig, ax = self._initialise_figure(fig=fig, ax=ax)
        self._plot(data, ax)

    def _plot(self, data: pd.DataFrame, ax: Axes) -> None:
        """Plot the daily view.

        Args:
            data: Prepared data for the daily view plot.
            ax: The matplotlib axis for the plot.
        """
        if len(data) == 1:
            start = data.index[0] - pd.Timedelta(days=1)
            end = data.index[0] + pd.Timedelta(days=1)
            data.reset_index().plot(
                ax=ax,
                kind="scatter",
                marker="o",
                x=data[self.config.column_label].name,
                y=self.config.column_label,
                xlim=(start, end),
                color=self.config.colour,
                label=self.config.column_label,
            )
            xticks, xticklabels = None, None
        else:
            data.plot(
                ax=ax,
                x=self.config.x_axis_label,
                y=self.config.column_label,
                label=self.config.column_label,
                color=self.config.colour,
            )
            xticklabels, xticks = self._generate_xticks(
                arr=list(data[self.config.x_axis_label].values),
                max_xticks=15,
                sep=self.config.string_separator,
            )

        self._set_axes_properties(
            ax=ax,
            properties={
                "title": self.config.translation_manager.translate("Daily Production"),
                "x_label": self.config.translation_manager.translate("Date"),
                "y_label": self.config.translation_manager.translate(
                    f"{' ('.join(self.config.column_label.split('_')) + ')'}"
                ).capitalize(),
                "xticks": xticks if xticks else None,
                "xticklabels": (
                    self.config.translation_manager.translate_list(
                        xticklabels, format_numbers=False
                    )
                    if xticklabels
                    else None
                ),
                "legend_visible": False,
            },
        )


def _get_plot_styles(
    plot_type: str,
    time_frame: str,
    translation_manager: TranslationManager,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get the plot styles based on the plot type and time frame.

    Args:
        plot_type: The type of plot style strategy. Valid options are 'rolling',
            'statistical', 'calendar'.
        time_frame: The time frame for the plot.
        translation_manager: Instance of TranslationManager to handle translations.
        **kwargs: Additional parameters required by the plot style strategy.

    Returns:
        A dictionary containing the plot styles for the specified plot type and
        time frame.
    """
    strategy = PlotStyleStrategy.get_strategy(plot_type)
    return strategy.get_plot_styles(time_frame, translation_manager, **kwargs)
