# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
Plot Manager Module.

This module provides the PlotManager class, which is designed to manage the
creation and layout of matplotlib figures and axes. It offers an organized and
extensible way to create complex plot layouts with both simple subplots and
advanced GridSpec layouts.

The PlotManager class allows users to:
- Create figures with specified rows and columns of subplots.
- Create figures with GridSpec layouts for advanced subplot arrangements.
- Retrieve specific axes for plotting.
- Update legends for figures based on specified axes and configurations.
- Apply predefined plot styles to the figures.
- Display and save all managed figures.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator, TypeGuard

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

TextModificationType = str | None
ReplaceLabelType = dict[str, str] | None
AdditionalItemsType = list[tuple[Any, str]] | None
AnyModificationType = AdditionalItemsType | TextModificationType | ReplaceLabelType
ModificationType = dict[str, AnyModificationType]


class PlotManager:
    """
    A class to manage the creation and layout of matplotlib figures and axes.

    Attributes:
        figures: A dictionary to store figure handles.
        axes: A dictionary to store axes handles.
        current_style_params: A dictionary to store the current style parameters.

    Methods:
        apply_plot_theme: Apply a predefined style to the plots.
        create_figure: Create a new figure with the specified number of rows
            and columns of subplots.
        create_multiple_figures: Create multiple figures with the specified
            parameters.
        create_gridspec_figure: Create a new figure with a GridSpec layout.
        create_multiple_gridspec_figures: Create multiple GridSpec figures with
            the specified parameters.
        update_legend: Update legend for a matplotlib figure based on specified
            axes and additional configurations.
        get_style_attribute: Retrieve a specific style attribute.
        get_all_style_attributes: Retrieve all current style attributes.
        get_axes: Retrieve the axes for a given figure and axis index.
        get_figure: Retrieve the figure handle for a given figure ID.
        show_all: Display all the figures managed by PlotManager.
        save_all: Save all the figures managed by PlotManager to the specified
            directory.
        manage_figure: Context manager to handle showing and optionally saving
            figures automatically.

    Example usage:
        plot_manager = PlotManager()

        # Create a simple figure with 1 row and 2 columns of subplots
        plot_manager.create_figure('fig1', nrows=1, ncols=2)

        # Retrieve the axes for the subplots and plot data
        ax1 = plot_manager.get_axes('fig1', 0)
        ax2 = plot_manager.get_axes('fig1', 1)

        ax1.plot([1, 2, 3], [4, 5, 6])
        ax1.set_title('Plot 1')

        ax2.plot([3, 2, 1], [6, 5, 4])
        ax2.set_title('Plot 2')

        # Display all figures
        plot_manager.show_all()

        # Save all figures to the 'plots' directory
        plot_manager.save_all('plots')

        # Create a figure with a GridSpec layout
        plot_manager.create_gridspec_figure('fig2', nrows=2, ncols=2,
            gridspec_kwargs=dict(height_ratios=[1, 2], width_ratios=[2, 1]))

        # Display all figures
        plot_manager.show_all()

        # Save all figures to the 'plots' directory
        plot_manager.save_all('plots')

        # Creating multiple figures without context manager
        fig_params = [
            {'fig_id': 'fig1', 'nrows': 1, 'ncols': 2, 'figsize': (10, 6)},
            {'fig_id': 'fig2', 'nrows': 2, 'ncols': 2, 'figsize': (12, 8)}
        ]
        plot_manager.create_multiple_figures(fig_params)

        # Using context manager without saving
        with plot_manager.manage_figure('fig3'):
            plot_manager.create_figure('fig3', nrows=1, ncols=2)
            ax1 = plot_manager.get_axes('fig3', 0)
            ax2 = plot_manager.get_axes('fig3', 1)

            ax1.plot([1, 2, 3], [4, 5, 6])
            ax1.set_title('Plot 1')

            ax2.plot([3, 2, 1], [6, 5, 4])
            ax2.set_title('Plot 2')
    """

    def __init__(self, theme: str = "frequenz-neustrom"):
        """Initialize a PlotManager instance and optionally apply a plot style.

        Args:
            theme: The name of the plot style to apply.
        """
        self.figures: dict[str, Figure] = {}
        self.axes: dict[str, list[Axes]] = {}
        self.current_style_params: dict[str, Any] = {}

        # Register a custom colormap for Frequenz-Neustrom
        self._freqstrom_cmap_name = "freqstrom"
        self._freqstrom_cmap_primary_colour = "#000000"
        try:
            mpl.colormaps[self._freqstrom_cmap_name]
        except KeyError:
            my_cmap = ListedColormap(
                [
                    self._freqstrom_cmap_primary_colour,
                    "#00FFCE",
                    "#600DFF",
                    "#F42784",
                    "#00AEEF",
                    "#FFC200",
                    "#1A1A1A",
                    "#393939",
                    "#585858",
                    "#777777",
                    "#979797",
                    "#B6B6B6",
                    "#D5D5D5",
                    "#F4F4F4",
                ]
            )
            mpl.colormaps.register(cmap=my_cmap, name=self._freqstrom_cmap_name)

        self.apply_plot_theme(theme)
        logger.info("PlotManager initialised with theme: %s", theme)

    def apply_plot_theme(self, theme: str) -> None:
        """Apply a predefined style to the plots.

        Args:
            theme: The name of the plot theme to apply.

        Raises:
            ValueError: If the specified theme is not recognized
        """
        themes: dict[str, dict[str, Any]] = {
            "frequenz-neustrom": {
                "base": {"base_theme": "seaborn-v0_8-white"},
                "params": {
                    "axes.edgecolor": "0.8",
                    "axes.facecolor": "white",
                    "axes.grid": False,
                    "axes.grid.which": "both",
                    "axes.labelcolor": "black",
                    "axes.labelsize": 20,
                    "axes.linewidth": 1.5,
                    "axes.spines.bottom": True,
                    "axes.spines.left": True,
                    "axes.spines.right": False,
                    "axes.spines.top": False,
                    "axes.titlecolor": "black",
                    "axes.titlesize": 22,
                    "figure.dpi": 100,
                    "figure.figsize": (20, 12),
                    "figure.titlesize": 25,
                    "figure.titleweight": "bold",
                    "font.family": ["Roboto", "HelveticaNeue", "Arial", "sans-serif"],
                    "font.size": 14,
                    "grid.color": "0.8",
                    "grid.linestyle": "--",
                    "grid.linewidth": 0.8,
                    "image.cmap": self._freqstrom_cmap_name,
                    "legend.fontsize": 18,
                    "lines.color": self._freqstrom_cmap_primary_colour,
                    "lines.linewidth": 2,
                    "lines.marker": "o",
                    "lines.markersize": 6,
                    "savefig.format": "png",
                    "text.color": "black",
                    "xtick.color": "black",
                    "xtick.labelsize": 18,
                    "ytick.color": "black",
                    "ytick.labelsize": 18,
                },
            },
            "elegant-minimalist": {
                "base": {"base_theme": "seaborn-v0_8-whitegrid"},
                "params": {
                    "axes.edgecolor": "0.8",
                    "axes.facecolor": "white",
                    "axes.grid": True,
                    "axes.grid.which": "both",
                    "axes.labelcolor": "black",
                    "axes.labelsize": 14,
                    "axes.linewidth": 1.5,
                    "axes.spines.bottom": True,
                    "axes.spines.left": True,
                    "axes.spines.right": False,
                    "axes.spines.top": False,
                    "axes.titlecolor": "black",
                    "axes.titlesize": 16,
                    "figure.dpi": 100,
                    "figure.figsize": (20, 12),
                    "figure.titlesize": 19,
                    "figure.titleweight": "bold",
                    "font.family": "sans-serif",
                    "font.size": 14,
                    "grid.color": "0.8",
                    "grid.linestyle": "--",
                    "grid.linewidth": 0.8,
                    "image.cmap": "viridis",
                    "legend.fontsize": 12,
                    "lines.color": "black",
                    "lines.linewidth": 2,
                    "lines.marker": "o",
                    "lines.markersize": 6,
                    "savefig.format": "png",
                    "text.color": "black",
                    "xtick.color": "black",
                    "xtick.labelsize": 12,
                    "ytick.color": "black",
                    "ytick.labelsize": 12,
                },
            },
            "vibrant": {
                "base": {"base_theme": "seaborn-v0_8-bright"},
                "params": {
                    "axes.axisbelow": True,
                    "axes.edgecolor": "gray",
                    "axes.facecolor": "white",
                    "axes.grid": True,
                    "axes.labelcolor": "black",
                    "axes.titlesize": 16,
                    "figure.facecolor": "white",
                    "figure.figsize": (20, 12),
                    "figure.titlesize": 19,
                    "figure.titleweight": "bold",
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Comic Sans MS"],
                    "font.size": 14,
                    "grid.color": "silver",
                    "grid.linestyle": "-",
                    "image.cmap": "turbo",
                    "legend.borderaxespad": 0.5,
                    "legend.edgecolor": "black",
                    "legend.facecolor": "white",
                    "legend.fontsize": 12,
                    "legend.frameon": True,
                    "legend.loc": "best",
                    "lines.color": "black",
                    "lines.linewidth": 3,
                    "text.color": "black",
                    "xtick.color": "black",
                    "ytick.color": "black",
                },
            },
            "classic": {
                "base": {"base_theme": "classic"},
                "params": {
                    "axes.edgecolor": "black",
                    "axes.facecolor": "white",
                    "axes.grid": False,
                    "axes.labelcolor": "black",
                    "axes.labelsize": 12,
                    "axes.linewidth": 1,
                    "axes.spines.bottom": True,
                    "axes.spines.left": True,
                    "axes.spines.right": True,
                    "axes.spines.top": True,
                    "axes.titlecolor": "black",
                    "axes.titlesize": 16,
                    "axes.titleweight": "normal",
                    "figure.dpi": 100,
                    "figure.figsize": (20, 12),
                    "figure.titlesize": 19,
                    "figure.titleweight": "bold",
                    "font.family": "serif",
                    "font.size": 12,
                    "grid.color": "grey",
                    "grid.linestyle": ":",
                    "grid.linewidth": 0.5,
                    "image.cmap": "viridis",
                    "legend.fontsize": 10,
                    "lines.color": "black",
                    "lines.linewidth": 1.5,
                    "lines.marker": "s",
                    "lines.markersize": 5,
                    "savefig.format": "png",
                    "text.color": "black",
                    "xtick.color": "black",
                    "xtick.labelsize": 10,
                    "ytick.color": "black",
                    "ytick.labelsize": 10,
                },
            },
        }
        if theme in themes:
            base_theme: str = themes[theme]["base"]["base_theme"]
            theme_params: dict[str, Any] = themes[theme]["params"]
            plt.style.use(base_theme)
            plt.rcParams.update(theme_params)
            self.current_style_params = theme_params
            logger.info("Applied theme: %s", theme)
        else:
            raise ValueError(f"Style '{theme}' is not recognized.")

    def create_figure(
        self,
        fig_id: str,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[int, int] = (10, 6),
    ) -> tuple[Figure, list[Axes]]:
        """Create a new figure with the specified number of rows and columns of subplots.

        Args:
            fig_id: Identifier for the figure.
            nrows: Number of rows of subplots.
            ncols: Number of columns of subplots.
            figsize: Size of the figure.

        Returns:
            The created figure and axes.

        Raises:
            ValueError:
                - If the figure with the given ID already exists.
                - If the number of rows or columns is less than 1.
        """
        if fig_id in self.figures:
            raise ValueError(f"Figure with id '{fig_id}' already exists.")
        if nrows < 1 or ncols < 1:
            raise ValueError("Number of rows and columns must be at least 1.")
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        self.figures[fig_id] = fig
        self.axes[fig_id] = [axs] if nrows * ncols == 1 else list(axs.flatten())
        logger.info(
            "Created figure '%s' with %s rows and %s columns.", fig_id, nrows, ncols
        )
        return fig, self.axes[fig_id]

    def create_multiple_figures(
        self,
        fig_params: (
            list[dict[str, str]]
            | list[dict[str, int]]
            | list[dict[str, tuple[int, int]]]
        ),
    ) -> None:
        """Create multiple figures with the specified parameters.

        Args:
            fig_params: List of dictionaries, each containing parameters for
                creating a figure.

        Example:
            fig_params = [
                {'fig_id': 'fig1', 'nrows': 1, 'ncols': 2, 'figsize': (10, 6)},
                {'fig_id': 'fig2', 'nrows': 2, 'ncols': 2, 'figsize': (12, 8)}
            ]
        """
        for params in fig_params:
            _, _ = self.create_figure(**params)

    # pylint: disable-next=too-many-arguments
    def create_gridspec_figure(
        self,
        *,
        fig_id: str,
        nrows: int,
        ncols: int,
        figsize: tuple[int, int] = (10, 6),
        gridspec_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Create a new figure with a GridSpec layout.

        Args:
            fig_id: Identifier for the figure.
            nrows: Number of rows in the GridSpec layout.
            ncols: Number of columns in the GridSpec layout.
            figsize: Size of the figure.
            gridspec_kwargs: Additional keyword arguments for GridSpec.

        Raises:
            ValueError: If the figure with the given ID already exists.
        """
        if fig_id in self.figures:
            raise ValueError(f"Figure with id '{fig_id}' already exists.")
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            nrows=nrows, ncols=ncols, **(gridspec_kwargs if gridspec_kwargs else {})
        )
        self.figures[fig_id] = fig
        self.axes[fig_id] = [
            fig.add_subplot(gs[row, col])
            for row in range(nrows)
            for col in range(ncols)
        ]
        logger.info(
            "Created GridSpec figure '%s' with %s rows and %s columns.",
            fig_id,
            nrows,
            ncols,
        )

    def create_multiple_gridspec_figures(
        self,
        fig_params: (
            list[dict[str, str]]
            | list[dict[str, int]]
            | list[dict[str, tuple[int, int]]]
            | list[dict[str, dict[str, Any]]]
        ),
    ) -> None:
        """Create multiple GridSpec figures with the specified parameters.

        Args:
            fig_params: List of dictionaries, each containing parameters for
                creating a GridSpec figure.

        Example:
            fig_params = [
                {
                    "fig_id": "fig1",
                    "nrows": 2,
                    "ncols": 2,
                    "figsize": (10, 6),
                    "gridspec_kwargs": {
                        "height_ratios": [1, 2],
                        "width_ratios": [2, 1],
                    }
                },
                {"fig_id": "fig2", "nrows": 3, "ncols": 3, "figsize": (15, 10)},
            ]
        """
        for params in fig_params:
            self.create_gridspec_figure(**params)

    def adjust_axes_spacing(self, fig_id: str, pixels: float = 100.0) -> None:
        """Adjust the spacing between axes to have specified pixel spacing.

        Args:
            fig_id: Identifier for the figure.
            pixels: The spacing between axes in pixels.

        Raises:
            ValueError: If the figure ID is not found.
        """
        if fig_id not in self.figures:
            raise ValueError(f"Figure '{fig_id}' does not exist.")

        fig = self.figures[fig_id]
        dpi = fig.dpi
        fig_width, fig_height = fig.get_size_inches()
        # Convert pixel spacing to relative units
        hspace = pixels / fig_height / dpi
        wspace = pixels / fig_width / dpi
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
        fig.tight_layout(rect=(0, 0, 1, 0.98))

    def update_legend(
        self,
        fig_id: str,
        axs: list[Axes],
        on: str = "axes",
        modifications: ModificationType | None = None,
        **legend_kwargs: Any,
    ) -> None:
        """Update legend for a matplotlib figure or its axes.

        Args:
            fig_id: Identifier for the figure.
            axs: A matplotlib Axes object or a list of Axes from which to collect
                handles and labels.
            on: Specify whether to update legends on 'axes' or 'figure'.
            modifications: A dictionary containing modifications:
                - 'additional_items': List of tuples (handle, label) to add to
                    legends. For on == 'axes', this should be a list of lists
                    corresponding to each axis. For on == 'figure', this should
                    be a list of tuples and all will be added to the figure
                    legend.
                - 'remove_label': A label to remove from legends.
                - 'replace_label': A dictionary mapping old labels to new labels.
            **legend_kwargs: Additional keyword arguments for the legend function.

        Raises:
            ValueError: If the figure ID is not found or if inputs are invalid.

        Example:
            modifications = {
                'additional_items': [
                    [(handle1, 'New Label 1')], [(handle2, 'New Label 2')]
                ],
                'remove_label': 'Old Label',
                'replace_label': {'Old Label': 'New Label'},
            }
            plot_manager.update_legend(
                'fig1', [ax1, ax2], on='axes',
                modifications=modifications, loc='upper right'
            )
        """
        if fig_id not in self.figures:
            raise ValueError(f"Figure '{fig_id}' does not exist.")

        fig = self.figures[fig_id]

        if isinstance(axs, Axes):
            axs = [axs]

        if modifications is None:
            modifications = {}

        fig.legends.clear()
        if on == "axes":
            self._update_axes_legends(list(axs), modifications, **legend_kwargs)
            logger.info("Updated legend for axes in figure '%s'.", fig_id)
        elif on == "figure":
            self._update_figure_legend(fig, list(axs), modifications, **legend_kwargs)
            logger.info("Updated legend for figure '%s'.", fig_id)
        else:
            raise ValueError(
                "Invalid value for 'on' parameter. Must be 'figure' or 'axes'."
            )
        fig.tight_layout()

    def _update_axes_legends(
        self,
        axs: list[Axes],
        modifications: ModificationType,
        **legend_kwargs: Any,
    ) -> None:
        """Update legends on individual axes.

        Args:
            axs: List of matplotlib Axes objects.
            modifications: A dictionary containing modifications.
            **legend_kwargs: Additional keyword arguments for the legend function.

        Raises:
            TypeError: If the inputs are invalid.
        """
        additional_items, remove_label, replace_label = self._get_modifications(
            modifications
        )
        if not self._is_additional_items_type(additional_items):
            raise TypeError("additional_items must be of type AdditionalItemsType")
        if not self._is_text_modification_type(remove_label):
            raise TypeError("remove_label must be of type TextModificationType")
        if not self._is_replace_label_type(replace_label):
            raise TypeError("replace_label must be of type ReplaceLabelType")

        for i, ax in enumerate(axs):
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().set_visible(False)

            axis_additional_items = (
                additional_items[i]
                if additional_items and i < len(additional_items)
                else None
            )
            if isinstance(axis_additional_items, tuple):
                if axis_additional_items[0] is None:
                    axis_additional_items = None

            handles, labels = self._process_handles_labels(
                handles, labels, remove_label, replace_label, axis_additional_items
            )
            ax.legend(handles, labels, **legend_kwargs)

    def _update_figure_legend(
        self,
        fig: Figure,
        axs: list[Axes],
        modifications: ModificationType,
        **legend_kwargs: Any,
    ) -> None:
        """Update legend on the figure.

        Args:
            fig: Matplotlib Figure object.
            axs: List of matplotlib Axes objects.
            modifications: A dictionary containing modifications.
            **legend_kwargs: Additional keyword arguments for the legend function.

        Raises:
            TypeError: If the inputs are invalid.
        """
        additional_items, remove_label, replace_label = self._get_modifications(
            modifications
        )
        if not self._is_additional_items_type(additional_items):
            raise TypeError("additional_items must be of type AdditionalItemsType")
        if not self._is_text_modification_type(remove_label):
            raise TypeError("remove_label must be of type TextModificationType")
        if not self._is_replace_label_type(replace_label):
            raise TypeError("replace_label must be of type ReplaceLabelType")

        all_handles, all_labels = [], []
        for ax in axs:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().set_visible(False)

            handles, labels = self._process_handles_labels(
                handles, labels, remove_label, replace_label, None
            )
            all_handles.extend(handles)
            all_labels.extend(labels)

        if additional_items:
            extra_handles, extra_labels = zip(
                *[
                    (handle, label)
                    for handle, label in additional_items
                    if handle is not None
                ]
            )
            all_handles.extend(extra_handles)
            all_labels.extend(extra_labels)

        fig.legend(all_handles, all_labels, **legend_kwargs)

    @staticmethod
    def _process_handles_labels(
        handles: list[Any],
        labels: list[str],
        remove_label: TextModificationType,
        replace_label: ReplaceLabelType,
        additional_items: tuple[Any, str] | None,
    ) -> tuple[list[Any], list[str]]:
        """Process handles and labels by removing, replacing, and adding items.

        Args:
            handles: List of handles.
            labels: List of labels.
            remove_label: Label to remove from the legend.
            replace_label: Dictionary mapping old labels to new labels.
            additional_items: Tuple of handle and label to add to the legend.

        Returns:
            Tuple of processed handles and labels.

        Raises:
            ValueError: If the lengths of handles and labels do not match.
        """
        if len(handles) != len(labels):
            raise ValueError("Handles and labels must be of the same length.")

        if remove_label:
            handles_labels = [
                (h, l) for h, l in zip(handles, labels) if l != remove_label
            ]
            _handles, _labels = zip(*handles_labels) if handles_labels else ([], [])
            handles, labels = list(_handles), list(_labels)

        if replace_label:
            labels = [replace_label.get(old_label, old_label) for old_label in labels]

        if additional_items:
            try:
                handles.append(additional_items[0])
                labels.append(additional_items[1])
            except IndexError as e:
                logger.warning("Failed to add additional items to legend: %s", e)
        return list(handles), list(labels)

    def _get_modifications(
        self,
        modifications: ModificationType,
    ) -> tuple[AnyModificationType, AnyModificationType, AnyModificationType]:
        """Get modifications from the input dictionary.

        Args:
            modifications: A dictionary containing modifications.

        Returns:
            tuple of additional items, remove label, and replace label.
        """
        additional_items = modifications.get("additional_items", None)
        remove_label = modifications.get("remove_label", None)
        replace_label = modifications.get("replace_label", None)
        return additional_items, remove_label, replace_label

    @staticmethod
    def _is_additional_items_type(
        obj: AnyModificationType,
    ) -> TypeGuard[AdditionalItemsType]:
        """Check if the object is of type AdditionalItemsType.

        Args:
            obj: The object to check.

        Returns:
            True if the object is of type AdditionalItemsType, False otherwise.
        """
        if obj is None:
            return True
        if isinstance(obj, list):
            return all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str)
                for item in obj
            )
        return False

    @staticmethod
    def _is_text_modification_type(
        obj: AnyModificationType,
    ) -> TypeGuard[TextModificationType]:
        """Check if the object is of type TextModificationType.

        Args:
            obj: The object to check.

        Returns:
            True if the object is of type TextModificationType, False otherwise.
        """
        return obj is None or isinstance(obj, str)

    @staticmethod
    def _is_replace_label_type(obj: AnyModificationType) -> TypeGuard[ReplaceLabelType]:
        """Check if the object is of type ReplaceLabelType.

        Args:
            obj: The object to check.

        Returns:
            True if the object is of type ReplaceLabelType, False otherwise.
        """
        return obj is None or (
            isinstance(obj, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items())
        )

    def get_style_attribute(self, attribute: str) -> Any:
        """Retrieve a specific style attribute.

        Args:
            attribute: The name of the style attribute to retrieve.

        Returns:
            The value of the requested style attribute, or None if it does not
            exist.
        """
        return self.current_style_params.get(attribute, None)

    def get_all_style_attributes(self) -> dict[str, str | int | float]:
        """Retrieve all current style attributes.

        Returns:
            A dictionary of all current style attributes.
        """
        return self.current_style_params

    def get_axes(self, fig_id: str, ax_idx: int | None = None) -> list[Axes]:
        """Retrieve the axes for a given figure and axis index.

        Args:
            fig_id: Identifier for the figure.
            ax_idx: Index of the axis to retrieve. If None, return all axes.

        Returns:
            The requested axis or axes as a list.

        Raises:
            ValueError: If the figure does not exist.
            IndexError: If the axis index is out of bounds.
            TypeError: If the axes are not stored in a list.
        """
        if fig_id in self.axes:
            axes = self.axes[fig_id]
            if ax_idx is None:
                return axes
            if isinstance(axes, list):
                if ax_idx < len(axes):
                    return [axes[ax_idx]]
                raise IndexError(
                    f"Axis index '{ax_idx}' is out of bounds for figure '{fig_id}'."
                )
            raise TypeError(f"Axes for figure '{fig_id}' are not stored in a list.")
        raise ValueError(f"Figure '{fig_id}' does not exist.")

    def get_figure(self, fig_id: str) -> Figure:
        """Retrieve the figure handle for a given figure ID.

        Args:
            fig_id: Identifier for the figure.

        Returns:
            The requested figure.

        Raises:
            ValueError: If the figure does not exist.
        """
        if fig_id in self.figures:
            return self.figures[fig_id]
        raise ValueError(f"Figure '{fig_id}' does not exist.")

    def show_all(self) -> None:
        """Display all the figures managed by PlotManager.

        Raises:
            RuntimeError: If there are no figures to display.
        """
        if not self.figures:
            raise RuntimeError("No figures to display.")

        for fig in self.figures.values():
            fig.tight_layout()
            fig.show()
        logger.info("Displayed all figures.")

    def save_all(self, directory: str) -> None:
        """Save all the figures managed by PlotManager to the specified directory.

        Args:
            directory: Directory to save the figures.

        Raises:
            ValueError: If the directory is not specified.
        """
        if not directory:
            raise ValueError("Directory not specified.")

        if not self.figures:
            logger.info("No figures to save.")

        os.makedirs(directory, exist_ok=True)
        for fig_id, fig in self.figures.items():
            fig.savefig(f"{directory}/{fig_id}.png")
        logger.info("Saved all figures to directory %s", directory)

    @contextmanager
    def manage_figure(
        self, fig_id: str, save: bool = False, directory: str | None = None
    ) -> Generator[None, None, None]:
        """Context manager to handle showing and optionally saving figures automatically.

        Args:
            fig_id: Identifier for the figure.
            save: Whether to save the figures after showing them.
            directory: Directory to save the figures if save is True.

        Yields:
            None: The context within which the figure is managed.

        Raises:
            ValueError: If save is True and directory is not specified.
        """
        try:
            yield
        finally:
            self.show_all()
            if save:
                if directory:
                    self.save_all(directory)
                else:
                    raise ValueError("Directory must be specified if save is True.")
            logger.info(
                "Managed figure '%s' with automatic display and optional save.", fig_id
            )
