# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
Module for interacting with the weather API service for fetching historical forecast data.

The module provides a client for the weather API service. The client can be used
to retrieve historical weather forecast data for multiple locations and a given
time range.
"""

from datetime import datetime

import grpc.aio as grpcaio
import pandas as pd
from frequenz.client.weather._client import Client
from frequenz.client.weather._types import ForecastFeature, Location


async def fetch_historical_weather_forecasts(  # pylint: disable=too-many-arguments
    *,
    service_address: str,
    feature_names: list[str],
    locations: list[tuple[float, float]],
    start_time: datetime,
    end_time: datetime,
    file_to_store: str = "",
) -> pd.DataFrame:
    """Fetch historical weather forecast data and return a pandas dataframe.

    Args:
        service_address: The address of the service to connect to given in a
            form of a host followed by a colon and a port.
        feature_names: The list of forecast feature names. Each feature is a
            string representing a ForecastFeature enum value.
        locations: The list of locations to retrieve the forecast data for.
            Expects location as a tuple of (latitude, longitude) in this order.
        start_time: Start of the time range to get weather forecasts for.
        end_time: End of the time range to get weather forecasts for.
        file_to_store: The filename to optionally store the data. The data
            will be stored in the specified file format. Supported formats
            are 'csv' and 'parquet'.

    Returns:
        A pandas dataframe containing the historical weather forecast data.

    Raises:
        ValueError: If the file format is not supported.
    """
    client = Client(service_address)

    features = [ForecastFeature[fn] for fn in feature_names]
    locations_in = [
        Location(latitude=lat, longitude=lon, country_code="")
        for (lat, lon) in locations
    ]

    location_forecast_iterator = client.hist_forecast_iterator(
        features=features, locations=locations_in, start=start_time, end=end_time
    )

    # The try and except block was added as a work-around until fixed in the service.
    # NOTE: Remove when fixed and uncomment the code block below.
    rows = []
    try:
        async for forecasts in location_forecast_iterator:
            rows.extend(forecasts.flatten())
    except grpcaio.AioRpcError:
        # this error occurs when forecasts for multiple locations are requested
        # can be ignored
        pass
    # rows = [
    #     item
    #     async for forecasts in location_forecast_iterator
    #     for item in forecasts.flatten()
    # ]

    df = pd.DataFrame(rows)
    valid_file_formats = ["csv", "parquet"]
    if file_to_store:
        file_format = file_to_store.split(".")[-1].lower()
        if file_format == "parquet":
            df.to_parquet(file_to_store)
        elif file_format == "csv":
            df.to_csv(file_to_store)
        else:
            raise ValueError(
                f"Unsupported file format: {file_format}. "
                f"Supported formats are: {', '.join(valid_file_formats)}."
            )
    return df
