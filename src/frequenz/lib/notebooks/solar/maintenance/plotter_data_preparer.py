# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
Module for preparing data for various plot types.

This module provides classes for preparing the data for plotting as well as
generating production statistics. Each class is designed to work with a specific
plot type from the `plotter` module and uses configurations defined in the
`plotter_config` module.

Classes:
    - `BasePreparer`: Abstract base class for data preparers.
    - `CalendarPreparer`: Prepares data for calendar view plots.
    - `RollingPreparer`: Prepares data for rolling view plots.
    - `ProfilePreparer`: Prepares data for statistical profile plots.
    - `DailyPreparer`: Prepares data for daily production plots.
    - `StatsPreparer`: Generates production statistics.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd

from frequenz.lib.notebooks.solar.maintenance.data_processing import segment_and_analyse
from frequenz.lib.notebooks.solar.maintenance.plotter_config import (
    CalendarViewConfig,
    DailyViewConfig,
    ProfileViewConfig,
    RollingViewConfig,
    StatsViewConfig,
)


class BasePreparer(ABC):
    """Abstract base class for preparing data for plotting."""

    def __init__(self, config: Any):
        """Initialise the data preparer with the configuration.

        Args:
            config: The data preparation configuration.
        """
        self.config = config
        self._date_format = (
            f"%d{self.config.string_separator}%b{self.config.string_separator}%Y"
        )
        self._time_format = "%H:%M"

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Prepare the data for plotting.

        Args:
            df: The input data to prepare.

        Returns:
            A DataFrame with the prepared data.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement the prepare method.")


class CalendarPreparer(BasePreparer):
    """Data preparer for calendar view plots."""

    def __init__(self, config: CalendarViewConfig) -> None:
        """Initialise the data preparer with the configuration.

        Args:
            config: The data preparation configuration for calendar view plots.
        """
        super().__init__(config)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for calendar view plots.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A DataFrame with the prepared data for the calendar view plot.
        """
        _df = df.copy(deep=True)
        return self._prepare_data(_df)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for the calendar view plot.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A DataFrame with the prepared data for the calendar view.

        Raises:
            ValueError: If an invalid time frame is provided.
        """
        current = df.index[-1]
        time_frame_mapping: dict[str, Callable[[], pd.DataFrame]] = {
            "day": lambda: df.loc[
                current.replace(hour=0, minute=0, second=0) : current
            ],
            "month": lambda: df.loc[
                current.replace(day=1, hour=0, minute=0, second=0) : current
            ]
            .resample("D")
            .sum(),
            "year": lambda: df.loc[
                current.replace(month=1, day=1, hour=0, minute=0, second=0) : current
            ]
            .resample("ME")
            .sum(),
            "12month": lambda: df.loc[
                current.replace(day=1, hour=0, minute=0, second=0)
                - pd.DateOffset(years=1) : current
            ]
            .resample("ME")
            .sum(),
            "all": lambda: df.resample("YE").sum(),
        }
        if self.config.time_frame not in time_frame_mapping:
            raise ValueError(f"Invalid time frame: {self.config.time_frame}")
        return time_frame_mapping[self.config.time_frame]()


class RollingPreparer(BasePreparer):
    """Data preparer for rolling view plots."""

    def __init__(self, config: RollingViewConfig) -> None:
        """Initialise the data preparer with the configuration.

        Args:
            config: The data preparation configuration for rolling view plots.
        """
        super().__init__(config)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for rolling view plots.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A DataFrame with the prepared data for the rolling view plot.
        """
        _df = df.copy(deep=True)
        return self._prepare_data(_df)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for the rolling view plot.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A DataFrame with the prepared data for the rolling view plot.

        Raises:
            ValueError: If an invalid time frame is provided.
        """
        time, time_frame = self.config.view[0], self.config.view[1].lower()
        if (not isinstance(time, int)) or (time <= 0):
            raise ValueError("Time must be a positive integer.")

        if (time_frame == "days") & (time == 1):
            time, time_frame = 24, "hours"

        if self.config.rolling_average:
            df_to_plot = self._process_rolling_average(
                df, time, "D" if time_frame == "days" else "h"
            )
        else:
            df_to_plot = self._prepare_data_time_frame(df, time, time_frame)

        is_empty = (
            df_to_plot.empty
            if not self.config.rolling_average
            else df_to_plot.count().empty
        )
        if is_empty:
            return df_to_plot
        df_to_plot.columns = df_to_plot.columns.astype(str)
        return df_to_plot

    def _process_rolling_average(
        self, df: pd.DataFrame, time: int, resample_freq: str
    ) -> pd.DataFrame:
        """Process the rolling average for the rolling view plot.

        Args:
            df: DataFrame with solar power production data and a datetime index.
            time: The time window for the rolling average.
            resample_freq: The frequency for resampling the data.

        Returns:
            A DataFrame with the processed rolling average data.
        """
        df = df.resample(resample_freq).sum()
        # Drop leap day if present (for consistency in yearly data)
        df.drop(
            df[
                (pd.to_datetime(df.index).month == 2)
                & (pd.to_datetime(df.index).day == 29)
            ].index,
            inplace=True,
        )
        df = df.rolling(window=time, min_periods=time).mean()
        df[self.config.x_axis_label] = [
            item.strftime(self._date_format) for item in df.index
        ]
        return df

    def _prepare_data_time_frame(
        self, df: pd.DataFrame, time: int, time_frame: str
    ) -> pd.DataFrame:
        """Prepare the data for the rolling view plot based on the time frame.

        Args:
            df: DataFrame with solar power production data and a datetime index.
            time: The time window for the rolling view.
            time_frame: The time frame for the rolling view.

        Returns:
            A DataFrame with the prepared data for the rolling view plot.

        Raises:
            ValueError: If an invalid time frame is provided.
        """
        current = df.index[-1]
        if time_frame == "days":
            start = current.replace(hour=0, minute=0, second=0) - pd.Timedelta(
                days=time
            )
            stop = current.replace(
                hour=23, minute=59, second=59, microsecond=1
            ) - pd.Timedelta(days=1)
            df_to_plot = df.loc[start:stop, :].resample("D").sum()
            df_to_plot[self.config.x_axis_label] = [
                item.strftime(self._date_format) for item in df_to_plot.index
            ]
        elif time_frame == "hours":
            df_to_plot = df
            df_to_plot[self.config.x_axis_label] = [
                item.time().strftime(self._time_format)
                for item in pd.to_datetime(df_to_plot.index)
            ]
            start = current - pd.Timedelta(hours=time) + pd.Timedelta(microseconds=1)
            stop = current
            df_to_plot = df_to_plot.loc[start:stop, :]
        else:
            raise ValueError("Invalid time frame.")
        return df_to_plot


class ProfilePreparer(BasePreparer):
    """Data preparer for statistical production profile plots."""

    def __init__(self, config: ProfileViewConfig) -> None:
        """Initialise the data preparer with the configuration.

        Args:
            config: The data preparation configuration for statistical view plots.
        """
        super().__init__(config)

    def prepare(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Prepare data for statistical production profile plots.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A dictionary containing the prepared data for the statistical
            production profile plots.
        """
        _df = df.copy(deep=True)
        return self._prepare_data(_df)

    def _prepare_data(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Prepare data for the statistical production profile plot.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A dictionary containing the prepared data for the statistical
            production profile plots. The dictionary keys are the groupings
            provided in the configuration. The values are DataFrames with the
            prepared data for each grouping.

        Raises:
            ValueError: If an invalid grouping is provided.
            RuntimeError: If the number of axes does not match the number of
                groupings. This is a sanity check and should not (normally) occur.
        """
        current = df.index[-1]
        past_n_days = df.loc[
            current.replace(hour=0, minute=0, second=0)
            - pd.Timedelta(days=self.config.duration) : current.replace(
                hour=23, minute=59, second=59, microsecond=1
            )
            - pd.Timedelta(days=1)
        ]

        supported_groupings = {
            "grouped": (list(pd.DatetimeIndex(past_n_days.index).time), None, False),
            "continuous": (None, None, False),
            "24h_continuous": ("D", "D", True),
        }

        if any(group not in supported_groupings for group in self.config.groupings):
            raise ValueError("Invalid grouping provided.")

        user_groupings = {
            group: supported_groupings[group] for group in self.config.groupings
        }
        past_n_days_stats = segment_and_analyse(
            data=past_n_days,
            grouping_freq_list=[item for item, _, _ in user_groupings.values()],
            resamp_freq_list=[item for _, item, _ in user_groupings.values()],
            group_labels=list(user_groupings.keys()),
            exclude_zeros=[item for _, _, item in user_groupings.values()],
            verbose=False,  # fix to False because this will be replaced by a logger
        )
        for group, stats in past_n_days_stats.items():
            if group == "grouped":
                try:
                    x_axis = [
                        f"{item.strftime(self._time_format)}" for item in stats.index
                    ]
                    current_time_formatted = current.strftime(self._time_format)
                except AttributeError:
                    x_axis = [str(item) for item in stats.index]
                    current_time_formatted = str(current.hour)
                stats[self.config.x_axis_label] = x_axis
                stats = stats.reindex(
                    index=np.roll(
                        stats.index,
                        -np.where(
                            stats[self.config.x_axis_label] == current_time_formatted
                        )[0][0]
                        - 1,
                    )
                )
            elif group == "continuous":
                stats[self.config.x_axis_label] = [
                    f"{item.strftime(self._date_format)}{self.config.string_separator}"
                    f"({item.strftime(self._time_format)})"
                    for item in stats.index
                ]
            elif group == "24h_continuous":
                stats[self.config.x_axis_label] = [
                    item.strftime(self._date_format) for item in stats.index
                ]
            past_n_days_stats[group] = stats
        if len(self.config.groupings) != len(past_n_days_stats.keys()):
            raise RuntimeError(
                "Number of requested groupings does not match the number of returned axes."
            )
        return past_n_days_stats


class DailyPreparer(BasePreparer):
    """Data preparer for daily production plots."""

    def __init__(self, config: DailyViewConfig) -> None:
        """Initialise the data preparer with the configuration.

        Args:
            config: The data preparation configuration for daily view plots.
        """
        super().__init__(config)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for daily production plots.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A DataFrame with the prepared data for the daily production plot.
        """
        _df = df.copy(deep=True)
        return self._prepare_data(_df)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for the daily production plot.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A DataFrame with the prepared data for the daily production plot.
        """
        df_to_plot = df.resample("D").sum()[[self.config.column_label]]
        if len(df_to_plot) > 1:
            df_to_plot[self.config.x_axis_label] = [
                item.strftime(self._date_format) for item in df_to_plot.index
            ]
        return df_to_plot


class StatsPreparer(BasePreparer):
    """Data preparer for production statistics."""

    def __init__(self, config: StatsViewConfig) -> None:
        """Initialise the data preparer with the configuration.

        Args:
            config: The data preparation configuration for production statistics.
        """
        super().__init__(config)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for production statistics.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A DataFrame with the prepared data for the production statistics.
        """
        return self._prepare_data(df)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistics on the solar power production.

        Args:
            df: DataFrame with solar power production data and a datetime index.

        Returns:
            A DataFrame containing the following statistics:
                - Current yield (kWh): The current energy (i.e. that of the latest
                    data point) energy in kilowatt-hours.
                - Yield today (kWh): The energy produced today in kilowatt-hours.
                - Yield this month (kWh): The energy produced this month (including
                    the current day) in kilowatt-hours.
                - Yield past 30 days (kWh): The energy produced in the past 30 days
                    (excluding the current day) in kilowatt-hours.
                - Yield this year (MWh): The energy produced this year (including
                    the current day) in megawatt-hours.
                - Yield past 365 days (MWh): The energy produced in the past 365
                    days (excluding the current day) in megawatt-hours.
                - Total yield (MWh): The energy produced in total in megawatt-hours.
        """
        current = df.index[-1]
        # pylint: disable=consider-using-f-string
        stats = pd.DataFrame(
            {
                self.config.translation_manager.translate("Current yield (kWh)"): [
                    self.config.translation_manager.translate(
                        "{:.2f}".format(df.loc[current, "energy_kWh"])
                    )
                ],
                self.config.translation_manager.translate("Yield today (kWh)"): [
                    self.config.translation_manager.translate(
                        "{:.2f}".format(
                            df[
                                pd.to_datetime(df.index).date == current.date()
                            ].energy_kWh.sum()
                        )
                    )
                ],
                self.config.translation_manager.translate("Yield this month (kWh)"): [
                    self.config.translation_manager.translate(
                        "{:.2f}".format(
                            df.loc[
                                current.replace(
                                    day=1, hour=0, minute=0, second=0
                                ) : current
                            ].energy_kWh.sum()
                        )
                    )
                ],
                self.config.translation_manager.translate(
                    "Yield past {value} days (kWh)", value=30
                ): [
                    self.config.translation_manager.translate(
                        "{:.2f}".format(
                            df.loc[
                                current.replace(hour=0, minute=0, second=0)
                                - pd.Timedelta(days=30) : current.replace(
                                    hour=23, minute=59, second=59
                                )
                                - pd.Timedelta(days=1)
                            ].energy_kWh.sum()
                        )
                    )
                ],
                self.config.translation_manager.translate("Yield this year (MWh)"): [
                    self.config.translation_manager.translate(
                        "{:.2f}".format(
                            df.loc[
                                current.replace(
                                    month=1, day=1, hour=0, minute=0, second=0
                                ) : current
                            ].energy_MWh.sum()
                        )
                    )
                ],
                self.config.translation_manager.translate(
                    "Yield past {value} days (MWh)", value=365
                ): [
                    self.config.translation_manager.translate(
                        "{:.2f}".format(
                            df.loc[
                                current.replace(hour=0, minute=0, second=0)
                                - pd.Timedelta(days=365) : current.replace(
                                    hour=23, minute=59, second=59
                                )
                                - pd.Timedelta(days=1)
                            ].energy_MWh.sum()
                        )
                    )
                ],
                self.config.translation_manager.translate("Total yield (MWh)"): [
                    self.config.translation_manager.translate(
                        "{:.2f}".format(df.energy_MWh.sum())
                    )
                ],
            },
            index=[self.config.translation_manager.translate("Energy production")],
        )
        return stats
