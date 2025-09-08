# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
Dedicated to data retrieval functionalities for the solar maintenance project.

This module provides functions that fetch and transform data from the weather
and reporting APIs or csv files.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from frequenz.client.common.metric import Metric
from frequenz.client.reporting import ReportingApiClient

from frequenz.datasci.weather.weather_api import fetch_historical_weather_forecasts
from frequenz.lib.notebooks.solar.maintenance.data_processing import (
    outlier_removal,
    preprocess_data,
    transform_weather_features,
)

_logger = logging.getLogger(__name__)


@dataclass
class BaseRetrievalConfig:
    """Base configuration for data retrieval."""

    service_address: str = field(
        metadata={
            "description": "API service address",
            "required": True,
        },
    )

    start_timestamp: datetime = field(
        default=datetime.now() - pd.Timedelta(days=1),
        metadata={
            "description": "Start timestamp for data retrieval",
        },
    )

    end_timestamp: datetime = field(
        default=datetime.now(),
        metadata={
            "description": "End timestamp for data retrieval",
        },
    )

    file_path: str = field(
        default="",
        metadata={
            "description": "Path to the file for file-based data retrieval",
            "validate": lambda x: x.endswith(".csv"),
        },
    )

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        if self.start_timestamp >= self.end_timestamp:
            raise ValueError("start_timestamp must be earlier than end_timestamp")


@dataclass(kw_only=True)
class WeatherRetrievalConfig(BaseRetrievalConfig):
    """Configuration for retrieving weather data."""

    feature_names: list[str] = field(
        metadata={
            "description": (
                "List of weather feature names to retrieve. "
                "Each feature is a string representing a ForecastFeature enum value",
            ),
            "required": True,
            "validate": lambda x: len(x) > 0,
        },
    )

    latitudes: list[float] = field(
        metadata={
            "description": "List of latitudes that must correspond to longitudes",
            "required": True,
            "validate": lambda x: len(x) > 0,
        },
    )

    longitudes: list[float] = field(
        metadata={
            "description": "List of longitudes that must correspond to latitudes",
            "required": True,
            "validate": lambda x: len(x) > 0,
        },
    )


@dataclass(kw_only=True)
class ReportingRetrievalConfig(BaseRetrievalConfig):
    """Configuration for retrieving reporting data."""

    api_key: str = field(
        metadata={
            "description": "API key for reporting service",
            "required": True,
        },
    )

    api_secret: str = field(
        metadata={
            "description": "API secret for reporting service",
            "required": True,
        },
    )

    microgrid_components: list[tuple[int, list[int]]] = field(
        metadata={
            "description": "List of microgrid-component ID pairs",
            "required": True,
        },
    )

    metrics_to_fetch: list[Metric] = field(
        metadata={
            "description": "List of reporting metrics to fetch",
            "required": True,
        },
    )

    resample_period_seconds: int = field(
        metadata={
            "description": "Resample period for the reporting api data in seconds",
            "required": True,
        },
    )


async def retrieve_data(
    config: WeatherRetrievalConfig | ReportingRetrievalConfig,
) -> pd.DataFrame:
    """Retrieve data from the weather or reporting API or from a file.

    Args:
        config: The configuration object for data retrieval containing all
            necessary parameters. See WeatherRetrievalConfig and
            ReportingRetrievalConfig.

    Returns:
        A pandas DataFrame containing the retrieved data.

    Raises:
        ValueError:
            - If no service address or file path provided for weather or
                reporting data.
            - If the file path does not end with '.csv'.
        TypeError: If an unknown configuration type is provided.
    """
    supported_configs = (WeatherRetrievalConfig, ReportingRetrievalConfig)
    if isinstance(config, WeatherRetrievalConfig):
        if config.service_address:
            _logger.info("Retrieving data from the weather API...")
            return await fetch_historical_weather_forecasts(
                service_address=config.service_address,
                feature_names=config.feature_names,
                locations=list(zip(config.latitudes, config.longitudes)),
                start_time=config.start_timestamp,
                end_time=config.end_timestamp,
            )
        if config.file_path:
            _logger.info("Retrieving weather data from file: %s", config.file_path)
            return pd.read_csv(
                config.file_path, parse_dates=["validity_ts"], index_col="validity_ts"
            ).reset_index()
        raise ValueError("No service address or file path provided for weather data.")
    if isinstance(config, ReportingRetrievalConfig):
        if config.service_address:
            _logger.info("Retrieving data from the reporting API...")
            reporting_client = ReportingApiClient(
                server_url=config.service_address,
                auth_key=config.api_key,
                sign_secret=config.api_secret,
            )
            reporting_data = [
                sample
                async for sample in reporting_client.receive_microgrid_components_data(
                    microgrid_components=config.microgrid_components,
                    metrics=config.metrics_to_fetch,
                    start_time=config.start_timestamp,
                    end_time=config.end_timestamp,
                    resampling_period=timedelta(seconds=config.resample_period_seconds),
                )
            ]
            return pd.DataFrame(reporting_data)
        if config.file_path:
            _logger.info("Retrieving reporting data from file: %s", config.file_path)
            return pd.read_csv(
                config.file_path, parse_dates=["ts"], index_col="ts"
            ).reset_index()
        raise ValueError("No service address or file path provided for reporting data.")
    raise TypeError(
        f"Unknown configuration type {type(config).__name__}. "
        f"Expected one of: {supported_configs}"
    )


def transform_weather_data(
    data: pd.DataFrame,
    weather_feature_names_mapping: dict[str, str],
    time_zone: ZoneInfo = ZoneInfo("UTC"),
) -> pd.DataFrame:
    """Transform weather forecast data.

    Args:
        data: The weather forecast data.
        weather_feature_names_mapping: Mapping of weather API feature names to
            internal feature names.
        time_zone: The timezone to convert the timestamps to. Should be a valid
            zoneinfo.ZoneInfo object.

    Returns:
        The transformed weather forecast data.

    Raises:
        ValueError: If missing or invalid date entries are found in 'validity_ts'.
    """
    if data.empty:
        _logger.info("No weather data available to transform.")
        return data
    _logger.info("Transforming weather forecast data...")
    weather_forecasts_df, nat_present = transform_weather_features(
        data=data,
        column_label_mapping=weather_feature_names_mapping,
        time_zone=time_zone,
    )
    if nat_present:
        raise ValueError(
            "Missing or invalid date entries found in 'validity_ts' column."
        )
    _logger.info("Weather forecast data transformed successfully.")
    return weather_forecasts_df


def transform_reporting_data(
    data: pd.DataFrame,
    microgrid_components: list[tuple[int, list[int]]] | None = None,
    outlier_detection_params: (
        dict[str, str | tuple[float, float] | dict[str, Any]] | None
    ) = None,
    time_zone: ZoneInfo = ZoneInfo("UTC"),
) -> pd.DataFrame:
    """Transform weather reporting data.

    Note: This wrapper function expects the timestamps to be in UTC and converts
    them to the provided timezone.

    Args:
        data: The reporting data.
        microgrid_components: List of microgrid-component ID pairs. If provided,
            the data is assumed to come from the reporting api and is subsequently
            pivoted and aggregated based on the microgrid components.
        outlier_detection_params: Dictionary of parameters for outlier detection.
        time_zone: The timezone to convert the timestamps to. Should be a valid
            zoneinfo.ZoneInfo object.

    Returns:
        The transformed reporting data.
    """
    if data.empty:
        _logger.info("No reporting data available to transform.")
        return data
    _logger.info("Transforming reporting data...")

    if microgrid_components:
        data, power_columns = _pivot_and_aggregate_data(data, microgrid_components)
    else:
        data.rename(columns={"p": "power", "ts": "timestamp"}, inplace=True)
        power_columns = {"power": ""}

    data = _convert_to_timezone(data, time_zone)

    if outlier_detection_params:
        data = _handle_outliers(
            data, outlier_detection_params, list(power_columns.keys())
        )

    client_data_df = preprocess_data(
        df=data,
        ts_col="timestamp",
        power_cols=list(power_columns.keys()),
        power_unit="kW",
        energy_units=["kWh", "MWh"],
        name_suffixes=list(power_columns.values()),
        datetime_format=None,
        in_place=True,
    )

    _logger.info("Reporting data transformed successfully.")
    return client_data_df


def _pivot_and_aggregate_data(
    data: pd.DataFrame, microgrid_components: list[tuple[int, list[int]]]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Pivot and aggregate data based on microgrid components.

    Args:
        data: The reporting data.
        microgrid_components: List of microgrid-component ID pairs.

    Returns:
        The pivoted and aggregated data and a dictionary of power columns.
    """
    data = data.pivot_table(
        index=["timestamp", "microgrid_id", "metric"],
        columns="component_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    data.columns.name = None
    power_columns = {}
    for mid, cids in microgrid_components:
        _cids = [cid for cid in cids if cid in data.columns]
        new_col = f"mid_{mid}_cids_" + "_".join([str(cid) for cid in _cids]) + "_SUM"
        power_columns[new_col] = f"mid{mid}"
        data[new_col] = data[_cids].sum(axis=1, skipna=True, min_count=1)
    return data, power_columns


def _convert_to_timezone(data: pd.DataFrame, time_zone: ZoneInfo) -> pd.DataFrame:
    """Convert timestamp to the given timezone.

    Timestamps are numpy.datetime64 (timezone-naive) and in UTC by default.

    Args:
        data: The reporting data.
        time_zone: The timezone to convert the timestamps to.

    Returns:
        The reporting data with the timestamp converted to the given timezone.
    """
    try:
        data["timestamp"] = pd.to_datetime(data["timestamp"].dt.tz_localize("UTC"))
    except TypeError:
        pass  # Already localized
    data["timestamp"] = pd.to_datetime(data["timestamp"].dt.tz_convert(time_zone))
    _logger.info("Timestamp column has been converted to timezone %s.", time_zone)
    return data


def _handle_outliers(
    data: pd.DataFrame,
    params: dict[str, str | tuple[float, float] | dict[str, Any]],
    power_column_labels: list[str],
) -> pd.DataFrame:
    """Handle outlier detection.

    Args:
        data: The reporting data.
        params: Dictionary of parameters for outlier detection.
        power_column_labels: List of power column labels.

    Returns:
        The reporting data after outlier detection and replacement.

    Raises:
        TypeError: If the bounds, method, or method_params are not of the
            expected type.
    """
    _logger.debug(
        "Columns that denote power values (will use these for outlier detection): %s",
        power_column_labels,
    )
    _logger.debug(
        "Reporting data before outlier detection: shape: %s\n%s",
        data.shape,
        data.head(),
    )

    bounds = params.get("bounds", (0.0, 0.0))
    method = params.get("method", "")
    method_params = params.get("params", {})
    if not isinstance(bounds, tuple):
        raise TypeError("bounds must be a tuple of (lower_bound, upper_bound)")
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    if not isinstance(method_params, dict):
        raise TypeError("method_params must be a dictionary")
    data = outlier_removal(
        data=data,
        columns=power_column_labels,
        bounds=bounds,
        method=method,
        **method_params,
    )

    _logger.debug(
        "Reporting data after outlier detection and replacement: shape: %s\n%s",
        data.shape,
        data.head(),
    )
    return data
