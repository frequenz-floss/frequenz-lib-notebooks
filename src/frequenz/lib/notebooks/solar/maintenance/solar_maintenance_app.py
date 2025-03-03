# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
This module contains the main entry point for the Solar Maintenance App.

The Solar Maintenance App is a tool that helps solar energy system operators
to monitor and maintain their solar energy systems. The app fetches and processes
weather and reporting data, generates production statistics, and plots the results.

The app is designed to be executed as a standalone application. The main entry
point for the Solar Maintenance App is the `run_workflow` function.
"""

# pylint: disable=too-many-lines

import datetime
import os
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from numpy.typing import NDArray
from pandas import Series
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from frequenz.lib.notebooks.solar.maintenance.config import SolarMaintenanceConfig
from frequenz.lib.notebooks.solar.maintenance.data_fetch import (
    ReportingRetrievalConfig,
    WeatherRetrievalConfig,
    retrieve_data,
    transform_reporting_data,
    transform_weather_data,
)
from frequenz.lib.notebooks.solar.maintenance.models import prepare_prediction_models
from frequenz.lib.notebooks.solar.maintenance.plot_manager import PlotManager
from frequenz.lib.notebooks.solar.maintenance.plot_styles import style_table
from frequenz.lib.notebooks.solar.maintenance.plotter import (
    DailyPlotter,
    ProfilePlotter,
    RollingPlotter,
)
from frequenz.lib.notebooks.solar.maintenance.plotter_config import (
    DailyViewConfig,
    ProfileViewConfig,
    RollingViewConfig,
    StatsViewConfig,
)
from frequenz.lib.notebooks.solar.maintenance.plotter_data_preparer import (
    DailyPreparer,
    ProfilePreparer,
    RollingPreparer,
    StatsPreparer,
)
from frequenz.lib.notebooks.solar.maintenance.translator import TranslationManager

Color = str | tuple[float, float, float] | tuple[float, float, float, float]
if TYPE_CHECKING:
    SeriesFloat = Series[float]
else:
    SeriesFloat = pd.Series  # Treated generically at runtime.


# pylint: disable=too-many-statements, too-many-branches, too-many-locals
async def run_workflow(
    user_config_changes: dict[str, Any],
) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
    """Run the Solar Maintenance App workflow.

    This function fetches and processes the necessary data, generates production
    statistics, and plots the results.

    Args:
        user_config_changes: A dictionary of user configuration changes.

    Returns:
        A dictionary containing the data for the plots and the production
        statistics table.

    Raises:
        ValueError:
            - If no API key is found in the .env file.
            - If the unit conversion of the data column for the short-term view
                to the column for the statistical profile view is not supported.
                This is not an issue in this version because the column labels
                (i.e. `short_term_view_col_to_plot` and
                `stat_profile_view_col_to_plot`) are hardcoded.
            - If the timezone of the data does not match the timezone in the
                configuration.
    """
    config, all_client_site_info = _load_and_validate_config(user_config_changes)

    load_dotenv(override=False)
    api_key = os.getenv("REPORTING_API_KEY")
    if api_key is None:
        raise ValueError(
            "No API key found. Please set the REPORTING_API_KEY in the .env file."
        )

    tm = TranslationManager(lang=config.language)

    list_of_latitudes, list_of_longitudes = [], []
    for _, v in all_client_site_info.items():
        list_of_latitudes += [v["latitude"]]
        list_of_longitudes += [v["longitude"]]
    weather_config = WeatherRetrievalConfig(
        service_address=config.weather_service_address,
        feature_names=list(config.weather_feature_names_mapping.keys()),
        latitudes=list_of_latitudes,
        longitudes=list_of_longitudes,
        start_timestamp=config.start_timestamp,
        end_timestamp=config.end_timestamp,
        verbose=config.verbose,
    )

    reporting_config = ReportingRetrievalConfig(
        service_address=config.reporting_service_address,
        api_key=api_key,
        microgrid_components=config.microgrid_components,
        metrics_to_fetch=config.metrics_to_fetch,
        resample_period_seconds=config.large_resample_period_seconds,
        start_timestamp=config.start_timestamp,
        end_timestamp=config.end_timestamp,
        verbose=config.verbose,
    )

    weather_data = await retrieve_data(weather_config)
    reporting_data = await retrieve_data(reporting_config)

    reporting_config.resample_period_seconds = config.small_resample_period_seconds
    reporting_config.start_timestamp = config.end_timestamp - datetime.timedelta(
        hours=config.real_time_view_duration_hours
    )
    if config.verbose:
        print(
            "Fetching data for shorter time resolution of "
            f"{config.small_resample_period_seconds}..."
        )
    reporting_data_higher_fs = await retrieve_data(reporting_config)

    weather_data = transform_weather_data(
        data=weather_data,
        weather_feature_names_mapping=config.weather_feature_names_mapping,
        time_zone=config.time_zone,
        verbose=config.verbose,
    )

    reporting_data = transform_reporting_data(
        data=reporting_data,
        microgrid_components=config.microgrid_components,
        outlier_detection_params=config.outlier_detection_parameters,
        time_zone=config.time_zone,
        verbose=config.verbose,
    )

    reporting_data_higher_fs = transform_reporting_data(
        data=reporting_data_higher_fs,
        microgrid_components=config.microgrid_components,
        outlier_detection_params=config.outlier_detection_parameters,
        time_zone=config.time_zone,
        verbose=config.verbose,
    )

    if config.force_positive_values:
        reporting_data = reporting_data.map(
            lambda x: abs(x) if np.issubdtype(type(x), np.number) else x
        )
        reporting_data_higher_fs = reporting_data_higher_fs.map(
            lambda x: abs(x) if np.issubdtype(type(x), np.number) else x
        )

    lat_lon_pairs = _create_lat_lon_pairs(
        weather_data["latitude"].unique(), weather_data["longitude"].unique()
    )

    # display the results for each microgrid separately
    production_legend_label = tm.translate("production")
    patch_label = tm.translate("current value")
    patch = None
    real_time_view_col_to_plot: list[Any] = ["power_kW"]
    real_time_view_ylabel = tm.translate(
        real_time_view_col_to_plot[0].replace("_", " (") + ")"
    ).capitalize()
    rolling_view_short_term_dur_hours = 24
    base_view_config_params = {
        "translation_manager": tm,
        "x_axis_label": "x-axis",
        "verbose": config.verbose,
    }
    # pylint: disable-next=too-many-nested-blocks
    for mid in reporting_data.microgrid_id.unique():
        message = f"Generating plots for microgrid ID: {mid}"
        if config.verbose:
            print(message)

        # NOTE: by convention, the columns are named with the microgrid ID inside data_fetch.py
        col_text = [f"_mid{mid}_"]
        data = _filter_and_rename_columns(
            reporting_data, col_text, verbose=config.verbose
        )
        if config.split_real_time_view_per_inverter:
            real_time_view_col_to_plot = [
                cid
                for components in config.microgrid_components
                if components[0] == mid
                for cid in components[1]
            ]
            data_higher_fs = _filter_and_rename_columns(
                reporting_data_higher_fs,
                real_time_view_col_to_plot,
                verbose=config.verbose,
            )
            # convert to kW; necessary because components are in raw values
            normalisation_factor = 1000
        else:
            data_higher_fs = _filter_and_rename_columns(
                reporting_data_higher_fs, col_text, verbose=config.verbose
            )
            normalisation_factor = 1
        timezone = str(pd.to_datetime(data.index).tzinfo)
        if timezone != config.time_zone.key:
            raise ValueError("Timezone mismatch.")

        pv_system = None
        if "simulation" in config.baseline_models:
            pv_system = _demo_pv_system_setup(
                all_client_site_info[mid], "Europe/Berlin", f"mid{mid}"
            )

        model_specs = _prepare_model_specs(config, all_client_site_info[mid], pv_system)
        prediction_models = prepare_prediction_models(
            data,
            model_specs,
            [k for k in config.baseline_models if k != "weather-based-forecast"],
        )
        if "weather-based-forecast" in config.baseline_models:
            closest_grid_point = _find_closest_grid_point(
                all_client_site_info[mid]["latitude"],
                all_client_site_info[mid]["longitude"],
                lat_lon_pairs,
            )
            prediction_models.update(
                prepare_prediction_models(
                    weather_data[
                        (weather_data["latitude"] == closest_grid_point[0])
                        & (weather_data["longitude"] == closest_grid_point[1])
                        & (
                            weather_data["validity_ts"]
                            <= config.end_timestamp + datetime.timedelta(hours=1)
                        )
                    ],
                    model_specs,
                    ["weather-based-forecast"],
                )
            )
        # NOTE: the below is a hack until PV lib simulation is properly set up
        # (i.e. needs user input for the PV system parameters)
        if "simulation" in prediction_models:
            prediction_models["simulation"]["predictions"] = (
                _post_process_simulation_results(
                    prediction_models["simulation"]["predictions"],
                    all_client_site_info[mid]["rated_power_watts"],
                    config.force_positive_values,
                )
            )

        # --- create the plot layout --- #
        plot_manager = PlotManager(theme=config.plot_theme)
        figures_and_axes, data_column_labels_to_plot, plot_settings = (
            _create_plot_layout(
                plot_manager=plot_manager,
                config=config,
                translation_manager=tm,
                timezone=timezone,
            )
        )
        short_term_view_col_to_plot = data_column_labels_to_plot["short_term_view"]
        long_term_view_col_to_plot = data_column_labels_to_plot["long_term_view"]
        stat_profile_view_col_to_plot = data_column_labels_to_plot["stat_profile_view"]

        common_rolling_view_config_params = {
            "primary_colour": plot_settings["primary_colour"],
            "cmap_name": plot_settings["colormap_name"],
            "rolling_average": False,
        }
        rolling_view_short_term_config = RollingViewConfig.create_config(
            base_view_config_params,
            common_rolling_view_config_params,
            view=(rolling_view_short_term_dur_hours, "hours"),
        )
        rolling_view_long_term_config = RollingViewConfig.create_config(
            base_view_config_params,
            common_rolling_view_config_params,
            view=(config.rolling_view_duration, config.rolling_view_time_frame),
        )
        rolling_view_average_config = RollingViewConfig.create_config(
            base_view_config_params,
            common_rolling_view_config_params,
            view=(config.rolling_view_duration, config.rolling_view_time_frame),
            rolling_average=True,
        )
        rolling_view_real_time_config = RollingViewConfig.create_config(
            base_view_config_params,
            common_rolling_view_config_params,
            view=(config.real_time_view_duration_hours, "hours"),
        )
        daily_plot_config = DailyViewConfig.create_config(
            base_view_config_params,
            column_label=long_term_view_col_to_plot,
            colour=plot_settings["primary_colour"],
        )
        statistical_plot_config = ProfileViewConfig.create_config(
            base_view_config_params,
            groupings=config.stat_profile_grouping,
            duration=config.rolling_view_duration,
            column_label=stat_profile_view_col_to_plot,
            cmap_name=plot_settings["colormap_name"],
        )
        stats_view_config = StatsViewConfig.create_config(base_view_config_params)
        # ------------------- #

        # --- prepare plot data --- #
        rolling_view_short_term = RollingPreparer(
            rolling_view_short_term_config
        ).prepare(data[[short_term_view_col_to_plot]])
        rolling_view_long_term = RollingPreparer(rolling_view_long_term_config).prepare(
            data[[long_term_view_col_to_plot]]
        )
        rolling_view_average = RollingPreparer(rolling_view_average_config).prepare(
            data[[long_term_view_col_to_plot]]
        )
        rolling_view_real_time = RollingPreparer(rolling_view_real_time_config).prepare(
            data_higher_fs[real_time_view_col_to_plot] / normalisation_factor
        )
        daily_production_view = DailyPreparer(daily_plot_config).prepare(data)
        statistical_view = ProfilePreparer(statistical_plot_config).prepare(data)
        # ------------------- #

        # --- generate the production statistics table --- #
        production_statistics_table_df = StatsPreparer(stats_view_config).prepare(data)
        style_table(production_statistics_table_df, show=True)
        # ------------------- #

        # --- short-term view --- #
        plotter_rolling_view_short_term = RollingPlotter(rolling_view_short_term_config)
        plotter_rolling_view_short_term.plot(
            data=rolling_view_short_term,
            fig=figures_and_axes["fig_short_term"]["figure"],
            ax=figures_and_axes["fig_short_term"]["axes"][0],
        )
        if plot_settings["show_annotation"]:
            recent_y = rolling_view_short_term[short_term_view_col_to_plot].iloc[-1]
            _annotate_last_point(
                figures_and_axes["fig_short_term"]["axes"][0],
                recent_y,
            )
            patch = Patch(color=plot_settings["patch_colour"], label=patch_label)
        figures_and_axes["fig_short_term"]["axes"][0].set_ylabel(
            figures_and_axes["fig_short_term"]["ylabel"]
        )
        # ------------------- #

        # --- long-term view --- #
        # rolling view
        plotter_rolling_view_long_term = RollingPlotter(rolling_view_long_term_config)
        plotter_rolling_view_long_term.plot(
            data=rolling_view_long_term,
            fig=figures_and_axes["fig_long_term"]["figure"],
            ax=figures_and_axes["fig_long_term"]["axes"][0],
        )
        if plot_settings["show_annotation"]:
            recent_y = rolling_view_long_term[long_term_view_col_to_plot].iloc[-1]
            _annotate_last_point(figures_and_axes["fig_long_term"]["axes"][0], recent_y)
            patch = Patch(color=plot_settings["patch_colour"], label=patch_label)
        figures_and_axes["fig_long_term"]["axes"][0].set_ylabel(
            figures_and_axes["fig_long_term"]["ylabel"]
        )

        # daily production
        plotter_daily_view = DailyPlotter(daily_plot_config)
        plotter_daily_view.plot(
            data=daily_production_view,
            fig=figures_and_axes["fig_long_term"]["figure"],
            ax=figures_and_axes["fig_long_term"]["axes"][2],
        )
        if plot_settings["show_annotation"]:
            recent_y = daily_production_view[long_term_view_col_to_plot].iloc[-1]
            _annotate_last_point(figures_and_axes["fig_long_term"]["axes"][2], recent_y)
            patch = Patch(color=plot_settings["patch_colour"], label=patch_label)

        # rolling view with rolling average
        plotter_rolling_view_average = RollingPlotter(rolling_view_average_config)
        plotter_rolling_view_average.plot(
            data=rolling_view_average,
            fig=figures_and_axes["fig_long_term"]["figure"],
            ax=figures_and_axes["fig_long_term"]["axes"][1],
        )
        if plot_settings["show_annotation"]:
            recent_y = rolling_view_average[long_term_view_col_to_plot].iloc[-1]
            _annotate_last_point(figures_and_axes["fig_long_term"]["axes"][1], recent_y)
            patch = Patch(color=plot_settings["patch_colour"], label=patch_label)
        figures_and_axes["fig_long_term"]["axes"][1].set_ylabel(
            figures_and_axes["fig_long_term"]["ylabel"]
        )
        # ------------------- #

        # --- real-time view --- #
        plotter_rolling_view_real_time = RollingPlotter(rolling_view_real_time_config)
        plotter_rolling_view_real_time.plot(
            data=rolling_view_real_time,
            fig=figures_and_axes["fig_real_time"]["figure"],
            ax=figures_and_axes["fig_real_time"]["axes"][0],
        )
        if plot_settings["show_annotation"]:
            if len(real_time_view_col_to_plot) == 1:
                for col in real_time_view_col_to_plot:
                    recent_y = rolling_view_real_time[str(col)].iloc[-2]
                    _annotate_last_point(
                        figures_and_axes["fig_real_time"]["axes"][0], recent_y
                    )
                    patch = Patch(
                        color=plot_settings["patch_colour"], label=patch_label
                    )
        figures_and_axes["fig_real_time"]["axes"][0].set_ylabel(real_time_view_ylabel)

        if plot_settings["legend_update_on"] == "figure":
            _legend_kwargs_copy = plot_settings["legend_kwargs"].copy()
            # divide legend labels into groups of 2 if needed
            _legend_kwargs_copy["ncol"] = max(
                _legend_kwargs_copy["ncol"],
                (
                    len(
                        figures_and_axes["fig_real_time"]["axes"][
                            0
                        ].get_legend_handles_labels()[1]
                    )
                    + 1
                )
                // 2,
            )
        else:
            _legend_kwargs_copy = plot_settings["legend_kwargs"]
        plot_manager.update_legend(
            fig_id="fig_real_time",
            axs=[figures_and_axes["fig_real_time"]["axes"][0]],
            on=plot_settings["legend_update_on"],
            modifications={
                "additional_items": (
                    [(patch, patch_label)] if plot_settings["show_annotation"] else None
                ),
                "replace_label": {
                    str(col): (
                        tm.translate("component_{value}", value=col)
                        if config.split_real_time_view_per_inverter
                        else production_legend_label
                    )
                    for col in real_time_view_col_to_plot
                },
            },
            **_legend_kwargs_copy,
        )
        # ------------------- #

        # --- plot the statistical production profile --- #
        plotter_profile_view = ProfilePlotter(statistical_plot_config)
        _ax_offset = 0
        for group_label, stats in statistical_view.items():
            if group_label == "grouped":
                _fig = figures_and_axes["fig_short_term"]["figure"]
                _ax = figures_and_axes["fig_short_term"]["axes"][1]
            else:
                _fig = figures_and_axes["fig_long_term"]["figure"]
                _ax = figures_and_axes["fig_long_term"]["axes"][3 + _ax_offset]
                _ax_offset += 1
            plotter_profile_view.plot(
                data=stats, fig=_fig, ax=_ax, group_label=group_label
            )
        if config.rolling_view_time_frame == "days":
            # --- overlay the short-term rolling view on the grouped stat plots --- #
            if (len(figures_and_axes["fig_short_term"]["axes"]) > 1) and any(
                ax.get_visible()
                for ax in figures_and_axes["fig_short_term"]["axes"][1:]
            ):
                overlay_label = tm.translate(
                    "production (past {value}h)",
                    value=rolling_view_short_term_dur_hours,
                )
                for stat_group in ["grouped"]:
                    idx = [
                        i
                        for i, e in enumerate(config.stat_profile_grouping)
                        if e == stat_group
                    ]
                    if idx:
                        _df = rolling_view_short_term.copy(deep=True)
                        # NOTE: the following only works for conversion between kW and kWh
                        if stat_profile_view_col_to_plot != short_term_view_col_to_plot:
                            if "power" in short_term_view_col_to_plot.lower():
                                _df[stat_profile_view_col_to_plot] = (
                                    _df[short_term_view_col_to_plot]
                                    * config.large_resample_period_seconds
                                    / 3600
                                )
                            elif "energy" in short_term_view_col_to_plot.lower():
                                _df[stat_profile_view_col_to_plot] = (
                                    _df[short_term_view_col_to_plot]
                                    / config.large_resample_period_seconds
                                    * 3600
                                )
                            else:
                                raise ValueError(
                                    f"Cannot convert {short_term_view_col_to_plot} to "
                                    f"{stat_profile_view_col_to_plot}"
                                )
                        ax = (
                            figures_and_axes["fig_short_term"]["axes"][
                                idx[0] + 1
                            ]  # + 1 to skip the first plot
                            if config.stat_profile_grouping
                            else figures_and_axes["fig_short_term"]["axes"][0]
                        )

                        ax.plot(
                            (
                                _df[base_view_config_params["x_axis_label"]]
                                if stat_group == "grouped"
                                else _df.index
                            ),
                            _df[stat_profile_view_col_to_plot],
                            "o--",
                            color=plot_settings["primary_colour"],
                            label=overlay_label,
                        )
                        statistical_view["grouped"][stat_profile_view_col_to_plot] = (
                            pd.Series(
                                data=_df[stat_profile_view_col_to_plot].values,
                                index=pd.to_datetime(_df.index).time,
                            )
                        )
            # ------------------- #
        # -------------------------------- #
        for i, (mdl_name, model_items) in enumerate(prediction_models.items()):
            n_models = len(prediction_models)
            cmap = plt.get_cmap(plt.rcParams["image.cmap"])
            cmap_values = np.linspace(0.1, 0.9, n_models)

            x_axis_short_term_view = rolling_view_short_term[
                [base_view_config_params["x_axis_label"]]
            ]
            x_axis_long_term_view = rolling_view_long_term[
                [base_view_config_params["x_axis_label"]]
            ]

            predictions_to_plot: list[pd.DataFrame] = []
            # predictions are shifted for plotting so that they do not contain the ground truth
            if model_specs[mdl_name]["target_label"] == short_term_view_col_to_plot:
                ax = [figures_and_axes["fig_short_term"]["axes"][0]]

                predictions = (
                    model_items["predictions"]
                    .shift(model_specs[mdl_name]["model_params"]["sampling_interval"])
                    .reindex(rolling_view_short_term.index, copy=False)
                    .to_frame()
                )
                predictions[base_view_config_params["x_axis_label"]] = (
                    x_axis_short_term_view
                )
                rolling_view_short_term[f"predictions_{mdl_name}"] = predictions[
                    "predictions"
                ]
                predictions_to_plot = [predictions]

            elif model_specs[mdl_name]["target_label"] == long_term_view_col_to_plot:
                ax = [figures_and_axes["fig_long_term"]["axes"][0]]

                predictions = (
                    model_items["predictions"]
                    .shift(1)
                    .reindex(rolling_view_long_term.index, copy=False)
                    .to_frame()
                )
                predictions[base_view_config_params["x_axis_label"]] = (
                    x_axis_long_term_view
                )
                rolling_view_long_term[f"predictions_{mdl_name}"] = predictions[
                    "predictions"
                ]
                predictions_to_plot = [predictions]

            else:
                ax = [
                    figures_and_axes["fig_short_term"]["axes"][0],
                    figures_and_axes["fig_long_term"]["axes"][0],
                ]

                predictions_1 = (
                    model_items["predictions"]
                    .shift(
                        int(
                            3600
                            * rolling_view_short_term_dur_hours
                            / config.large_resample_period_seconds
                        )
                    )
                    .reindex(rolling_view_short_term.index, copy=False)
                    .to_frame()
                )
                predictions_1[base_view_config_params["x_axis_label"]] = (
                    x_axis_short_term_view
                )
                rolling_view_short_term[f"predictions_{mdl_name}"] = predictions_1[
                    "predictions"
                ]

                predictions_2 = (
                    (
                        model_items["predictions"]
                        * config.large_resample_period_seconds
                        / 3600
                    )
                    .resample("D")
                    .sum()
                    .shift(1)
                    .reindex(rolling_view_long_term.index, copy=False)
                    .to_frame()
                )
                predictions_2[base_view_config_params["x_axis_label"]] = (
                    x_axis_long_term_view
                )
                rolling_view_long_term[f"predictions_{mdl_name}"] = predictions_2[
                    "predictions"
                ]

                predictions_to_plot = [predictions_1, predictions_2]

            for _ax, preds in zip(ax, predictions_to_plot):
                current_xlabel = _ax.get_xlabel()
                current_ylabel = _ax.get_ylabel()
                custom_xtick_labels = [
                    tick.get_text() for tick in _ax.get_xticklabels()
                ]
                preds.plot(
                    ax=_ax,
                    x=base_view_config_params["x_axis_label"],
                    y="predictions",
                    style="" if mdl_name == "simulation" else "s--",
                    kind="area" if mdl_name == "simulation" else "line",
                    color=(
                        cmap(cmap.N - 1)
                        if mdl_name == "simulation"
                        else cmap(cmap_values[i])
                    ),
                    label=tm.translate(mdl_name),
                    legend=False,
                    alpha=1 if mdl_name == "simulation" else 0.7,
                    zorder=0 if mdl_name == "simulation" else 2,
                )
                _ax.set_xticklabels(custom_xtick_labels)
                _ax.set_xlabel(current_xlabel)
                _ax.set_ylabel(current_ylabel)

        # --- update the figure legends --- #
        _ax = (
            figures_and_axes["fig_short_term"]["axes"][:2]
            if plot_settings["legend_update_on"] == "figure"
            else figures_and_axes["fig_short_term"]["axes"]
        )
        plot_manager.update_legend(
            fig_id="fig_short_term",
            axs=_ax,
            on=plot_settings["legend_update_on"],
            modifications={
                "additional_items": (
                    (
                        [(patch, patch_label)]
                        if plot_settings["legend_update_on"] == "figure"
                        else [(patch, patch_label)] + [(None, "")] * (len(_ax) - 1)
                    )
                    if plot_settings["show_annotation"]
                    else None
                ),
                "remove_label": (
                    short_term_view_col_to_plot
                    if plot_settings["legend_update_on"] == "figure"
                    else None
                ),
                "replace_label": (
                    {short_term_view_col_to_plot: production_legend_label}
                    if plot_settings["legend_update_on"] == "axes"
                    else None
                ),
            },
            **plot_settings["legend_kwargs"],
        )

        if plot_settings["legend_update_on"] == "axes":
            _ax = figures_and_axes["fig_long_term"]["axes"]
        else:
            _ax = (
                figures_and_axes["fig_long_term"]["axes"][:2]
                if set(["continuous", "24h_continuous"]).isdisjoint(
                    set(config.stat_profile_grouping)
                )
                else list(figures_and_axes["fig_long_term"]["axes"][:2])
                + [figures_and_axes["fig_long_term"]["axes"][3]]
            )
        plot_manager.update_legend(
            fig_id="fig_long_term",
            axs=_ax,
            on=plot_settings["legend_update_on"],
            modifications={
                "additional_items": (
                    (
                        [(patch, patch_label)]
                        if plot_settings["legend_update_on"] == "figure"
                        else [(patch, patch_label)]
                        + [(None, "")] * (len(_ax) - 3)
                        + [(patch, patch_label)] * 2
                    )
                    if plot_settings["show_annotation"]
                    else None
                ),
                "replace_label": {long_term_view_col_to_plot: production_legend_label},
            },
            **plot_settings["legend_kwargs"],
        )
        # -------------------------------- #
        for fig in figures_and_axes.keys():
            plot_manager.adjust_axes_spacing(fig_id=fig, pixels=100.0)

    return {
        "real_time_view": rolling_view_real_time,
        "rolling_view_short_term": rolling_view_short_term,
        "rolling_view_long_term": rolling_view_long_term,
        "rolling_view_average": rolling_view_average,
        "daily_production": daily_production_view,
        "statistical_profiles": statistical_view,
        "production_statistics_table": production_statistics_table_df,
    }


def _load_and_validate_config(
    user_config_changes: dict[str, Any],
) -> tuple[SolarMaintenanceConfig, dict[int, Any]]:
    """Load and validate configuration settings for the Solar Maintenance app.

    Args:
        user_config_changes: Dictionary containing user-provided config changes.

    Returns:
        A tuple containing the validated configuration object and a dictionary
        of all client site information.

    Raises:
        ValueError:
            - If 'microgrid_ids' and 'component_ids' are not provided.
            - If the number of client site information entries does not match
                the number of microgrid IDs.
    """
    config = SolarMaintenanceConfig()

    if {"microgrid_ids", "component_ids"}.issubset(user_config_changes):
        config.update_mids_and_cids(
            user_config_changes["microgrid_ids"],
            user_config_changes["component_ids"],
        )
    elif (
        "microgrid_ids" in user_config_changes or "component_ids" in user_config_changes
    ):
        raise ValueError(
            "Both 'microgrid_ids' and 'component_ids' must be provided together."
        )

    for param, value in user_config_changes.items():
        if param not in {"client_site_info", "microgrid_ids", "component_ids"}:
            config.update_parameter(param, value)

    if "client_site_info" in user_config_changes:
        if len(user_config_changes["client_site_info"]) != len(config.microgrid_ids):
            raise ValueError(
                "The number of client site information entries must match the "
                "number of microgrid IDs."
            )
    for idx, mid in enumerate(config.microgrid_ids):
        site_info = config.client_site_info[0].copy()
        if "client_site_info" in user_config_changes:
            config.update_dict(
                site_info,
                user_config_changes["client_site_info"][idx],
                "client_site_info",
            )
        config.client_site_info.append(site_info)
    config.client_site_info = config.client_site_info[-len(config.microgrid_ids) :]
    all_client_site_info = {
        mid: config.client_site_info[idx]
        for idx, mid in enumerate(config.microgrid_ids)
    }
    if config.verbose:
        print(f"Configuration parameters: {config.__dict__}")
    return config, all_client_site_info


def _create_plot_layout(  # pylint: disable=too-many-arguments
    *,
    plot_manager: PlotManager,
    config: SolarMaintenanceConfig,
    translation_manager: TranslationManager,
    timezone: str,
    legend_update_on: str = "axes",
    show_annotation: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Create and configure the plot layout for the Solar Maintenance App.

    Args:
        plot_manager: The PlotManager object.
        config: The configuration object.
        translation_manager: The TranslationManager for translating labels.
        timezone: The timezone of the client.
        legend_update_on: The location to update the legend. Accepts 'axes' or
            'figure'.
        show_annotation: If True, the last point in the plot is annotated.

    Returns:
        A tuple containing the PlotManager object and a dictionary with figures
        and axes.

    Raises:
        ValueError: If 'legend_update_on' is not 'axes' or 'figure'.
    """
    if legend_update_on not in ["axes", "figure"]:
        raise ValueError(
            "Invalid value for 'legend_update_on'. Expected 'axes' or 'figure'."
        )

    colormap_name = plot_manager.get_style_attribute("image.cmap")
    plot_settings = {
        "colormap_name": colormap_name,
        "primary_colour": plot_manager.get_style_attribute("lines.color"),
        "patch_colour": plt.get_cmap(colormap_name)(-1),
        "legend_update_on": legend_update_on,
        "legend_kwargs": {
            "figure": {  # ensure legend is placed at the bottom of the figure
                "bbox_to_anchor": (0.5, -0.03),
                "loc": "lower center",
                "ncol": 4,  # ensure a minimum of 4 columns in the legend
            },
            "axes": {  # ensure legend is placed at the right of and outside the plot
                "bbox_to_anchor": (1, 0.5),
                "loc": "center left",
                "ncol": 1,
            },
        }[legend_update_on],
        "show_annotation": show_annotation,
    }

    data_column_labels_to_plot = {
        "short_term_view": "power_kW",
        "long_term_view": "energy_kWh",
        "stat_profile_view": "energy_kWh",
    }

    subplots_short_term = 1 + int("grouped" in config.stat_profile_grouping)
    subplots_long_term = 3 + sum(
        item in config.stat_profile_grouping
        for item in ["continuous", "24h_continuous"]
    )

    fig_size_base = plot_manager.get_style_attribute("figure.figsize")
    plot_layout_specs = {
        "fig_real_time": {
            "nrows": 1,
            "ncols": 1,
            "figsize": fig_size_base,
            "title": translation_manager.translate(
                "Real-time View (All times are in {value})",
                value=translation_manager.translate(timezone),
            ),
            "ylabel": None,
        },
        "fig_short_term": {
            "nrows": subplots_short_term,
            "ncols": 1,
            "figsize": (fig_size_base[0], fig_size_base[1] + 2.5 * subplots_short_term),
            "title": translation_manager.translate(
                "Short-term View (All times are in {value})",
                value=translation_manager.translate(timezone),
            ),
            "ylabel": translation_manager.translate(
                data_column_labels_to_plot["short_term_view"].replace("_", " (") + ")"
            ).capitalize(),
        },
        "fig_long_term": {
            "nrows": subplots_long_term,
            "ncols": 1,
            "figsize": (fig_size_base[0], fig_size_base[1] + 2.5 * subplots_long_term),
            "title": translation_manager.translate(
                "Long-term View (All times are in {value})",
                value=translation_manager.translate(timezone),
            ),
            "ylabel": translation_manager.translate(
                data_column_labels_to_plot["long_term_view"].replace("_", " (") + ")"
            ).capitalize(),
        },
    }

    plot_manager.create_multiple_figures(
        [
            {
                "fig_id": fig_id,
                **{k: v for k, v in specs.items() if k not in ["title", "ylabel"]},
            }
            for fig_id, specs in plot_layout_specs.items()
        ]
    )

    figures_and_axes = {
        fig_id: {
            "figure": plot_manager.get_figure(fig_id),
            "axes": plot_manager.get_axes(fig_id),
            "title": specs["title"],
            "ylabel": specs["ylabel"],
        }
        for fig_id, specs in plot_layout_specs.items()
    }
    for fig_info in figures_and_axes.values():
        fig_info["figure"].suptitle(fig_info["title"])
    return figures_and_axes, data_column_labels_to_plot, plot_settings


def _filter_and_rename_columns(
    data: pd.DataFrame, substrings: list[str], verbose: bool = False
) -> pd.DataFrame:
    """Filter and rename columns in the data.

    Details:
        - Filters out any columns that do not contain the substrings in
            'substrings'. Non-string columns are converted to strings before
            checking. Note that the substrings are case-sensitive.
        - Renames the columns by removing all characters between the first and
          last underscore.
        - Removes any rows with all NaN values.

    Args:
        data: The input data.
        substrings: The list of substrings to filter the columns with.
        verbose: If True, the function prints messages to the console.

    Returns:
        The filtered and renamed data.
    """
    data = data[
        [
            col
            for col in data.columns
            if any(str(_col_text) in str(col) for _col_text in substrings)
        ]
    ].copy()
    data.dropna(how="all", inplace=True)
    rename_cols = {
        col: f"{col.split('_')[0]}_{col.split('_')[-1]}"
        for col in data.columns
        if isinstance(col, str)
    }
    data.rename(
        columns=rename_cols,
        inplace=True,
    )
    message = f"Renamed columns: {rename_cols}"
    if verbose:
        print(message)
    return data


def _annotate_last_point(
    ax: Axes, recent_y: float, patch_colour: Color | None = "lightgray"
) -> None:
    """Annotate the last point in the plot.

    Args:
        ax: The matplotlib axis to plot the data.
        recent_y: The y-value of the most recent data point.
        patch_colour: The colour for the annotation patch.
    """
    ax.annotate(
        f"{recent_y:.2f}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        xytext=(-20, -20),
        textcoords="offset points",
        bbox={
            "boxstyle": "round,pad=0.3",
            "edgecolor": "none",
            "facecolor": patch_colour,
        },
    )


def _demo_pv_system_setup(
    client_site_info: dict[str, Any], client_timezone: str, name: str
) -> dict[str, Any]:
    """Set up a demo PV system for the simulation.

    NOTE: Define the PV system (these parameters should be defined by the user)

    Args:
        client_site_info: The client site information.
        client_timezone: The client timezone.
        name: The name of the PV system.

    Returns:
        The PV system parameters for the simulation.
    """
    surface_tilt = 20
    sandia_modules = pvlib.pvsystem.retrieve_sam("SandiaMod")
    cec_inverters = pvlib.pvsystem.retrieve_sam("CECinverter")
    module = sandia_modules["Canadian_Solar_CS5P_220M___2009_"]
    inverter_parameters = cec_inverters["PV_Powered__PVP1100EVR__120V_"]
    temperature_parameters = TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ]

    location_parameters = {
        "latitude": client_site_info["latitude"],
        "longitude": client_site_info["longitude"],
        "altitude": client_site_info["altitude"],
        "timezone": client_timezone,
        "name": name,
    }
    array_1_parameters = {
        "module": module,
        "strings": 1,
        "modules_per_string": 3,
        "surface_tilt": surface_tilt,
        "surface_azimuth": 90,
        "temperature_parameters": temperature_parameters,
        "name": "East Facing Array",
    }
    array_2_parameters = {
        "module": module,
        "strings": 1,
        "modules_per_string": 3,
        "surface_tilt": surface_tilt,
        "surface_azimuth": 270,
        "temperature_parameters": temperature_parameters,
        "name": "West Facing Array",
    }
    pv_system_arrays = [array_1_parameters, array_2_parameters]
    return {
        "pv_system_arrays": pv_system_arrays,
        "location_parameters": location_parameters,
        "inverter_parameters": inverter_parameters,
    }


def _prepare_model_specs(
    config: SolarMaintenanceConfig,
    client_site_info: dict[str, Any],
    pv_system: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Prepare model specifications.

    Args:
        config: SolarMaintenanceConfig object.
        client_site_info: Contains the client site information. The dictionary
            should have the following keys: 'efficiency', 'peak_power_watts',
            and 'rated_power_watts' for the weather-based forecast model.
        pv_system: Containts the PV system arrays, location parameters, and
            inverter parameters for the simulation.

    Returns:
        A dictionary of all prepared model specifications.

    Raises:
        ValueError: If the PV system parameters are not provided for the simulation.
    """
    model_specs: dict[str, dict[str, Any]] = {}
    if "7-day MA" in config.baseline_models:
        model_specs.update(
            {
                "7-day MA": {
                    "model": "wma",
                    "target_label": "energy_kWh",
                    "resample_params": ("D", "sum"),
                    "model_params": {"mode": "uniform", "win_size": 7},
                }
            }
        )
    if "7-day sampled MA" in config.baseline_models:
        model_specs.update(
            {
                "7-day sampled MA": {
                    "model": "sampled_ma",
                    "target_label": "power_kW",
                    "resample_params": None,
                    "model_params": {
                        "window": pd.Timedelta(days=7),
                        "sampling_interval": (24 * 3600)
                        // config.large_resample_period_seconds,
                    },
                }
            }
        )
    if "simulation" in config.baseline_models:
        if pv_system is None:
            raise ValueError(
                "PV system parameters must be provided for the simulation."
            )
        model_specs.update(
            {
                "simulation": {
                    "model": "pvlib",
                    "target_label": None,
                    "resample_params": None,
                    "model_params": {
                        "location_parameters": pv_system["location_parameters"],
                        "pv_system_arrays": pv_system["pv_system_arrays"],
                        "inverter_parameters": pv_system["inverter_parameters"],
                        "start_year": 2010,
                        "end_year": 2020,
                        "sampling_rate": f"{config.large_resample_period_seconds // 60}min",
                        "weather_option": "tmy",
                        "time_zone": config.time_zone,
                    },
                }
            }
        )
    if "weather-based-forecast" in config.baseline_models:
        model_specs.update(
            {
                "weather-based-forecast": {
                    "model": "naive_eff_irr2power",
                    "target_label": None,
                    "resample_params": None,
                    "model_params": {
                        "col_label": config.weather_feature_names_mapping[
                            "SURFACE_NET_SOLAR_RADIATION"
                        ],
                        "eff": client_site_info["efficiency"],
                        "peak_power_watts": client_site_info["peak_power_watts"],
                        "rated_power_watts": client_site_info["rated_power_watts"],
                        "resample_rate": f"{config.large_resample_period_seconds}s",
                    },
                }
            }
        )
    return model_specs


def _create_lat_lon_pairs(
    latitudes: NDArray[np.float64], longitudes: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Create latitude and longitude pairs.

    Args:
        latitudes: The list of latitudes.
        longitudes: The list of longitudes.

    Returns:
        The latitude and longitude pairs.
    """
    grid1, grid2 = np.meshgrid(latitudes, longitudes, indexing="ij")
    lat_lon_pairs = np.column_stack((grid1.flatten(), grid2.flatten()))
    return lat_lon_pairs


def _find_closest_grid_point(
    client_latitude: float, client_longitude: float, lat_lon_pairs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Find the closest geographical grid point to the client's location.

    Args:
        client_latitude: The client's latitude.
        client_longitude: The client's longitude.
        lat_lon_pairs: The latitude and longitude pairs.

    Returns:
        The closest grid point to the client's location.
    """
    closest_grid_point = lat_lon_pairs[
        np.argmin(
            np.sqrt(
                np.sum(
                    np.square(lat_lon_pairs - [client_latitude, client_longitude]),
                    axis=1,
                )
            )
        ),
        :,
    ]
    return closest_grid_point


def _post_process_simulation_results(
    predictions: SeriesFloat,
    scale_value: float = 1.0,
    force_positive_values: bool = False,
) -> SeriesFloat:
    """Post-process the simulation results.

    Args:
        predictions: The simulation predictions. Expects Series of floats.
        scale_value: The scaling value.
        force_positive_values: If True, the values are forced to be positive.

    Returns:
        The processed simulation predictions.
    """
    processed_predictions = (predictions / predictions.max()) * scale_value / 1000
    if not force_positive_values:
        processed_predictions = processed_predictions.map(
            lambda x: -abs(x) if np.issubdtype(type(x), np.number) else x
        )
    return processed_predictions.astype(float)
