# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Data fetching for asset optimization reporting."""

import logging
import os
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from frequenz.client.assets import AssetsApiClient

from frequenz.data.microgrid import MicrogridData, load_configs
from frequenz.lib.notebooks._credentials import resolve_credentials
from frequenz.lib.notebooks.dayahead import fetch_day_ahead_prices

_logger = logging.getLogger(__name__)


def _align_series_to_index(index: pd.Index, series: pd.Series) -> pd.Series:
    """Align a time series to a target datetime index using forward fill."""
    target_index = pd.DatetimeIndex(index)
    aligned = series.copy()
    aligned.index = pd.DatetimeIndex(aligned.index)

    if target_index.tz is not None and aligned.index.tz is None:
        aligned.index = aligned.index.tz_localize(target_index.tz)
    elif target_index.tz is None and aligned.index.tz is not None:
        aligned.index = aligned.index.tz_convert("UTC").tz_localize(None)
    elif target_index.tz is not None and aligned.index.tz is not None:
        aligned.index = aligned.index.tz_convert(target_index.tz)

    aligned = aligned.sort_index()
    return aligned.reindex(target_index, method="ffill")


def _load_dotenv_files(dotenv_path: str | Sequence[str]) -> None:
    """Load dotenv files in order, later files overriding earlier ones.

    Overriding matters in notebooks: without it a variable loaded for one microgrid
    survives in the kernel and shadows the file loaded for the next one.
    """
    paths = [dotenv_path] if isinstance(dotenv_path, str) else dotenv_path
    for path in paths:
        # load_dotenv hands the path to open(), which does not expand "~".
        expanded = os.path.expanduser(path)
        if not load_dotenv(dotenv_path=expanded, override=True):
            _logger.warning("No environment variables loaded from %s.", expanded)


async def init_microgrid_data(
    *,
    microgrid_config_files: str | Path | list[str | Path] | None = None,
    dotenv_path: str | Sequence[str] | None = None,
) -> MicrogridData:
    """Load MicrogridData instance using environment variables.

    Args:
        microgrid_config_files: Path, or paths, to microgrid configuration files.
        dotenv_path: Optional path, or paths, to environment variable files. They
            are loaded in order and override both earlier files and the current
            environment, so list the most specific file last.

    Returns:
        MicrogridData instance.

    Raises:
        ValueError: If required environment variables are not set.
    """
    if dotenv_path is not None:
        _load_dotenv_files(dotenv_path)

    service_address = os.environ["REPORTING_API_URL"]
    reporting_key, reporting_secret = resolve_credentials(os.environ, "REPORTING_API")
    assets_key, assets_secret = resolve_credentials(os.environ, "ASSETS_API")

    if not reporting_key or not reporting_secret:
        raise ValueError(
            "Reporting API credentials not configured. "
            "Set FREQUENZ_API_KEY/FREQUENZ_API_SECRET or "
            "REPORTING_API_KEY/REPORTING_API_SECRET."
        )

    assets_client = AssetsApiClient(
        os.environ["ASSETS_API_URL"],
        auth_key=assets_key,
        sign_secret=assets_secret,
    )
    mcfg = {}
    try:
        assets_config = await load_configs(
            default_files=microgrid_config_files,
            assets_client=assets_client,
        )
        mcfg = assets_config.microgrids
    except RuntimeError:
        _logger.warning("Could not run async formula loading in current context. ")

    return MicrogridData(
        server_url=service_address,
        auth_key=reporting_key,
        sign_secret=reporting_secret,
        microgrid_configs=mcfg,
    )


def merge_day_ahead_prices(
    df: pd.DataFrame,
    *,
    dayahead_api_key: str | None = None,
    dayahead_country_code: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> pd.DataFrame:
    """Merge ENTSO-E day-ahead prices into an existing battery-usecase dataframe.

    Args:
        df: Base dataframe to enrich with day-ahead prices.
        dayahead_api_key: ENTSO-E API key. If not provided, the value is read
            from the ``ENTSOE_API_KEY`` environment variable.
        dayahead_country_code: ENTSO-E country code.
        start_time: Optional explicit start time for the day-ahead query.
        end_time: Optional explicit end time for the day-ahead query.

    Returns:
        A copy of ``df`` with a ``day_ahead_price`` column aligned to the dataframe
        index.
    """
    if df.empty:
        return df.copy()

    start = start_time or pd.Timestamp(df.index.min()).to_pydatetime()
    end = end_time or pd.Timestamp(df.index.max()).to_pydatetime()

    if start_time is None and len(df.index) > 1:
        resolution = pd.Timestamp(df.index[1]) - pd.Timestamp(df.index[0])
        end = (pd.Timestamp(df.index.max()) + resolution).to_pydatetime()

    da_prices = fetch_day_ahead_prices(
        entsoe_key=dayahead_api_key,
        start=start,
        end=end,
        country_code=dayahead_country_code,
    )

    merged = df.copy()
    merged["day_ahead_price"] = _align_series_to_index(merged.index, da_prices)
    return merged


# pylint: disable=too-many-arguments
async def fetch_data(
    mdata: MicrogridData,
    *,
    component_types: tuple[str],
    mid: int,
    start_time: datetime,
    end_time: datetime,
    resampling_period: timedelta,
    splits: bool = False,
    fetch_soc: bool = False,
) -> pd.DataFrame:
    """
    Fetch data of a microgrid and processes it for plotting.

    Args:
        mdata: MicrogridData object to fetch data from.
        component_types: List of component types to fetch data for.
        mid: Microgrid ID.
        start_time: Start time for data fetching.
        end_time: End time for data fetching.
        resampling_period: Time resolution for data fetching.
        splits: Whether to split the data into positive and negative parts.
        fetch_soc: Whether to fetch state of charge (SOC) data.

    Returns:
        DataFrame containing the processed data.

    Raises:
        ValueError: If no data is found for the given microgrid and time range or if
            unexpected component types are present in the data.
    """
    _logger.info(
        "Requesting data from %s to %s at %s resolution",
        start_time,
        end_time,
        resampling_period,
    )
    df = await mdata.ac_active_power(
        microgrid_id=mid,
        component_types=component_types,
        start=start_time,
        end=end_time,
        resampling_period=resampling_period,
        keep_components=False,
        splits=splits,
        unit="kW",
    )
    if df is None or df.empty:
        raise ValueError(
            f"No data found for microgrid {mid} between {start_time} and {end_time}"
        )

    _logger.debug("Received %s rows and %s columns", df.shape[0], df.shape[1])

    if fetch_soc:
        soc_df = await mdata.soc(
            microgrid_id=mid,
            start=start_time,
            end=end_time,
            resampling_period=resampling_period,
            keep_components=False,
        )
        if soc_df is None or soc_df.empty:
            raise ValueError(
                f"No SOC data found for microgrid {mid} between {start_time} and {end_time}"
            )

        # Concat in case indices mismatch
        df = pd.concat([df, soc_df["battery"].rename("soc")], axis=1)

    # Default to nan for missing SOC data
    df["soc"] = df.get("soc", np.nan)

    # For later visualization we default to zero so we can use
    # the same plots for different microgrid setups
    df["battery"] = df.get("battery", 0)
    df["pv"] = df.get("pv", 0)
    df["chp"] = df.get("chp", 0)
    df["wind"] = df.get("wind", 0)

    # We only care about the generation part for this analysis
    df["pv"] = df["pv"].clip(upper=0)
    df["chp"] = df["chp"].clip(upper=0)
    df["wind"] = df["wind"].clip(upper=0)

    # Determine consumption if not present
    if "consumption" not in df.columns:
        cols = df.columns.tolist()
        if any(
            ct not in ["grid", "pv", "battery", "chp", "wind", "soc"] for ct in cols
        ):
            raise ValueError(
                f"Consumption not found in data and unexpected component types present: {cols}."
            )
        df["consumption"] = df["grid"] - (
            df["chp"] + df["pv"] + df["wind"] + df["battery"]
        )

    return df
