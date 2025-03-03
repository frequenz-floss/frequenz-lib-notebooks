# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
Handles all data processing and transformation tasks for the solar maintenance project.

The module contains functions to preprocess solar power production data, calculate
statistical metrics, segment and analyse the data, and transform weather features.
"""

import re
import warnings
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# pylint: disable-next=too-many-arguments, too-many-locals
def preprocess_data(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    power_cols: list[str] | tuple[str, ...] = ("p",),
    power_unit: str = "kW",
    energy_units: list[str] | tuple[str, ...] = ("kWh", "MWh"),
    name_suffixes: list[str] | tuple[str, ...] = ("midDefault",),
    datetime_format: str | None = None,
    in_place: bool = False,
) -> pd.DataFrame:
    """Preprocess by converting power to the required unit and calculating energy consumed.

    Details: The function converts the power column to the required unit and
    calculates the energy consumed based on the power and time difference
    between consecutive timestamps.

    Args:
        df: Input DataFrame.
        ts_col: The name of the timestamp column.
        power_cols: Power column names.
        power_unit: The unit to convert power into ('kW', 'MW', etc.).
        energy_units: Units to calculate energy ('kWh', 'MWh', etc.).
        name_suffixes: Suffixes to add to the power and the corresponding
            energy column names. The strings should be unique and descriptive
            (e.g. midX to reflect the microgrid ID X) and should match the
            length of power_cols.
        datetime_format: Optional datetime format if auto-parsing fails.
        in_place: Modify the DataFrame in-place or return a new DataFrame.

    Returns:
        The transformed DataFrame.

    Raises:
        ValueError:
            - If the length of power_cols and name_suffixes is not equal.
            - If the power unit, time unit, or unit prefix is invalid.
            - If the data index is not monotonically increasing
        KeyError: If required columns are missing in the input data.

    Example:
        >>> df = pd.read_csv('data.csv', parse_dates=['timestamp'])
        >>> processed_df = preprocess_data(df, ts_col='timestamp', power_cols=['power'])
        >>> print(processed_df.head())
    """
    valid_power_factor_units = {"W": 1, "kW": 1e3, "MW": 1e6, "GW": 1e9}
    valid_unit_prefixes = {" ": 1, "k": 1e3, "M": 1e6, "G": 1e9}
    valid_time_units = {"h": 3600, "min": 60, "sec": 1}

    def parse_unit(unit_str: str) -> tuple[float, str]:
        """Parse unit strings to extract their numerical factor based on the SI prefixes.

        Args:
            unit_str: A string representing the unit.

        Returns:
            A tuple with the numerical factor and the base unit.

        Raises:
            ValueError: If the unit prefix is invalid.
        """
        base_unit = unit_str.lstrip("".join(valid_unit_prefixes.keys()))
        factor = valid_unit_prefixes.get(unit_str[0], None)
        if factor is None:
            raise ValueError(
                f"Unexpected unit string: {unit_str[0]}. "
                f"Choose one of {valid_unit_prefixes.keys()}."
            )
        return factor, base_unit

    if len(power_cols) != len(name_suffixes):
        raise ValueError("Length of power_cols and name_suffixes must be equal.")

    if not in_place:
        df = df.copy()
        df.sort_index(inplace=True)

    required_columns = {ts_col}.union(set(power_cols))
    if not required_columns.issubset(df.columns):
        raise KeyError(
            f"Missing required columns: {required_columns - set(df.columns)}"
        )

    # Handle datetime conversion
    if datetime_format:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], format=datetime_format)
        except ValueError as e:
            raise ValueError(
                f"Error parsing datetime column '{ts_col}' "
                f"with format '{datetime_format}': {e}"
            ) from e
    df.set_index(ts_col, inplace=True)

    # Convert power to the required unit
    power_factor = valid_power_factor_units.get(power_unit, None)
    if power_factor is None:
        raise ValueError(
            f"Unexpected power unit: {power_unit}. "
            f"Choose one of {valid_power_factor_units.keys()}."
        )
    df[power_cols] /= power_factor
    df.rename(
        columns={
            col: f"power_{col_id}_{power_unit}"
            for col, col_id in zip(power_cols, name_suffixes)
        },
        inplace=True,
    )

    # Compute the energy consumed
    time_diff_seconds_arr = (
        df.index.to_series().diff().dt.total_seconds()
    )  # divide by 3600 for hours
    time_diff_seconds_series = time_diff_seconds_arr[
        time_diff_seconds_arr.notna() & (time_diff_seconds_arr != 0)
    ]
    if time_diff_seconds_series.nunique() == 1:
        time_diff_seconds = time_diff_seconds_series.values[0]
    else:
        warnings.warn(
            "Detected multiple unique time differences; this may indicate inconsistent "
            "timestamps. Falling back to the mode of the time differences."
        )
        time_diff_seconds = time_diff_seconds_series.mode()[0]
    for unit in energy_units:
        energy_factor, base_unit = parse_unit(unit)
        time_conversion_factor = valid_time_units.get(base_unit[1:], None)
        if time_conversion_factor is None:
            raise ValueError(
                f"Unexpected time unit: {base_unit[1:]}. "
                f"Choose one of {valid_time_units.keys()}."
            )
        for col_id in name_suffixes:
            df[f"energy_{col_id}_{unit}"] = (
                df[f"power_{col_id}_{power_unit}"] * time_diff_seconds
            ) / ((energy_factor / power_factor) * time_conversion_factor)

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Data index is not monotonically increasing. Please check input data."
        )

    return df


def calculate_stats(df: pd.DataFrame, exclude_zeros: bool = False) -> pd.DataFrame:
    """Calculate statistical metrics for a given DataFrame and resampling rule.

    Args:
        df: DataFrame with solar power production data and a datetime index.
        exclude_zeros: A boolean flag to exclude zero values from the calculation.

    Returns:
        A new DataFrame with the calculated statistics or a DataFrame with NaN
        values if the input data or the data after excluding zeros is empty.
    """

    def _calculate(data: pd.DataFrame) -> pd.DataFrame:
        stat_funcs: dict[str, Callable[[pd.DataFrame], Any]] = {
            "mean": lambda x: x.mean(),
            "median": lambda x: x.median(),
            "min": lambda x: x.min(),
            "max": lambda x: x.max(),
            "25th percentile": lambda x: x.quantile(0.25),
            "75th percentile": lambda x: x.quantile(0.75),
        }
        stats_df = pd.DataFrame()
        for stat_name, func in stat_funcs.items():
            result = func(data)
            if isinstance(result, pd.Series):
                result.name = "0"
                frame = result.to_frame().transpose()
            else:
                frame = result
            stats_df = pd.concat(
                [
                    stats_df,
                    frame.rename(
                        columns={k: f"{k}_{stat_name}" for k in frame.columns}
                    ),
                ],
                axis=1,
            )
        return stats_df

    data_to_use = df[(df != 0).any(axis=1)] if exclude_zeros else df
    if data_to_use.empty:
        columns = [
            f"{col}_{name}"
            for name in [
                "mean",
                "median",
                "min",
                "max",
                "25th percentile",
                "75th percentile",
            ]
            for col in df.columns
        ]
        return pd.DataFrame(np.nan, index=["0"], columns=columns)
    return _calculate(data_to_use)


# pylint: disable-next=too-many-arguments, too-many-locals
def segment_and_analyse(
    data: pd.DataFrame,
    *,
    grouping_freq_list: str | list[Any],
    resamp_freq_list: list[str | None],
    group_labels: list[str],
    exclude_zeros: list[bool],
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Process data by segmenting and calculating statistics for given frequencies.

    Args:
        data: DataFrame with a datetime index.
        grouping_freq_list: A list of elements that define how to group the data.
            The list elements that can be one of the following:
                - strings that correspond to a frequency string (e.g., '15min',
                  '1h', '1D')
                - list of strings that correspond to column labels
                - list of values of any one type (e.g. datetime.time).
            For more details see `grouping_freq` argument in segment_and_align
            function.
        resamp_freq_list: List of frequency strings (e.g., '15min', '1h', '1D')
            to resample the data. A list element can be also None, in which case
            the frequency is inferred from the data. Note that only up-sampling
            is supported. See segment_and_align function for more details.
        group_labels: List of labels to use in the output dictionary for the
            segment statistics for each grouping frequency.
        exclude_zeros: List of boolean flags to exclude zero values from the
            calculation for each grouping frequency.
        verbose: A boolean flag to print additional information.

    Returns:
        A dictionary with the segment statistics for each grouping frequency.
    """
    results = {freq: pd.DataFrame() for freq in group_labels}
    if data.empty:
        return results
    for label, grouping_freq, resamp_freq, no_zeros in zip(
        group_labels, grouping_freq_list, resamp_freq_list, exclude_zeros
    ):
        segmented_data = segment_and_align(data, grouping_freq, resamp_freq, verbose)
        stats_df = pd.DataFrame()
        df_index = []
        for group, segment in segmented_data.items():
            stats_df = pd.concat(
                [stats_df, calculate_stats(df=segment, exclude_zeros=no_zeros)]
            )
            df_index.append(group)
        stats_df["timestamp"] = df_index
        stats_df.set_index("timestamp", drop=True, inplace=True)
        stats_df.sort_index(inplace=True)
        results[label] = stats_df
    return results


def segment_and_align(
    data: pd.DataFrame,
    grouping_freq: str | list[Any] | None = None,
    resamp_freq: str | None = None,
    verbose: bool = False,
) -> dict[Any, pd.DataFrame]:
    """Segment the input data into periods based on the given frequency.

    Notes:
        - If the resampling frequency is higher than the inferred frequency, the
          data is resampled to the higher frequency. Otherwise, the data is used
          as is.
        - Linear interpolation is used to fill missing values when upsampling.
        - The function does not downsample the data but rather segments it based
          on the grouping frequency.

    Args:
        data: DataFrame with a datetime index.
        grouping_freq: Frequency string (e.g., '15min', '1h', '1D') to group
            the data. If a list is provided, the grouping is done based on the
            list elements. The elements can be either strings that correspond
            to column labels, or values of any one type (e.g. datetime.time).
            For the latter case, the length of the list must be equal to that of
            the selected axis (index by default) and the values are used as-is
            to determine the groups. For more details see the pandas.groupby
            documentation.
        resamp_freq: Frequency string (e.g., '15min', '1h', '1D') to resample
            the data. If None, the frequency is inferred from the data and used.
        verbose: A boolean flag to print additional information.

    Returns:
        A dictionary with the segmented data for each period.
    """
    # ensure data has a datetime index
    data_index = pd.DatetimeIndex(data.index)
    data.set_index(data_index, inplace=True)

    inferred_freq = pd.infer_freq(data_index)
    if inferred_freq is None:
        inferred_freq = (
            f"{int(data_index.to_series().diff().mode()[0].total_seconds() / 60)}min"
        )
        message = (
            "pandas cannot infer the frequency automatically. "
            f"Found to be {inferred_freq} using the mode of the time differences."
        )
        if verbose:
            print(message)
    if grouping_freq is None:
        grouping_freq = inferred_freq
    if resamp_freq is None:
        resamp_freq = inferred_freq

    resampled_df = (
        data.resample(resamp_freq).interpolate(method="linear")
        if is_more_frequent(resamp_freq, inferred_freq)
        else data
    )

    grouper: str | list[Any] | pd.Grouper
    if isinstance(grouping_freq, list):
        grouper = grouping_freq
    else:
        grouper = pd.Grouper(freq=grouping_freq)

    return {
        period: group
        for period, group in resampled_df.groupby(grouper)
        if not group.empty
    }


def is_more_frequent(freq1: str, freq2: str) -> bool:
    """Compare two frequency strings to determine if the first is more frequent.

    Args:
        freq1: Frequency string (e.g., '15min')
        freq2: Frequency string (e.g., '10min')

    Returns:
        True if freq1 is more frequent than freq2, False otherwise.

    assert is_more_frequent("15min", "10min") == False
    assert is_more_frequent("30min", "1h") == True
    assert is_more_frequent("1D", "24h") == False
    assert is_more_frequent("1h", "1h") == False
    assert is_more_frequent("1h", "60min") == False
    """

    def convert_to_timedelta(freq: str) -> pd.Timedelta:
        """Convert a frequency string to a pandas Timedelta object."""
        return pd.to_timedelta(freq if re.match(r"^\d", freq) else "1" + freq)

    return convert_to_timedelta(freq1) < convert_to_timedelta(freq2)


def transform_weather_features(
    data: pd.DataFrame,
    column_label_mapping: dict[str, str],
    time_zone: ZoneInfo = ZoneInfo("UTC"),
    verbose: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Transform weather data by mapping features to new columns.

    Features are mapped to new columns with values from 'value' column. Creates
    a new column to show the time difference between 'validity_ts' and
    'creation_ts'. Expects the time columns to be in UTC and converts them to
    the provided timezone.

    Args:
        data: DataFrame with weather data.
        column_label_mapping: Dictionary that maps 'feature' entries to new
            column names with row entries obtained from the corresponding row
            of the 'value' column.
        time_zone: The timezone to convert the time columns to. Should be a
            valid zoneinfo.ZoneInfo object.
        verbose: A boolean flag to print additional information.

    Returns:
        A tuple of the transformed DataFrame and a boolean flag indicating if
        any missing or invalid date entries were found in 'validity_ts'.

    Raises:
        ValueError: If required columns are missing in the input data.
    """
    data_out = data.copy(deep=True)

    required_columns = ["feature", "value", "validity_ts", "creation_ts"]
    missing_columns = [col for col in required_columns if col not in data_out.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    data_out["feature_name"] = data_out["feature"].apply(
        lambda x: column_label_mapping.get(x.name, x.name)
    )
    data_out = data_out.pivot_table(
        index=["creation_ts", "latitude", "longitude", "validity_ts"],
        columns="feature_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    data_out.columns.name = None

    data_out["validity_ts"] = pd.to_datetime(
        data_out["validity_ts"], errors="coerce"
    ).astype("datetime64[us]")
    nat_present = data_out["validity_ts"].isna().any()
    if nat_present:
        message = (
            "Missing or invalid date entries found in 'validity_ts'. "
            "Handle the data accordingly."
        )
        if verbose:
            warnings.warn(message)
    # note: timestamps are numpy.datetime64 (timezone-naive) and in UTC by default
    try:
        data_out["creation_ts"] = pd.to_datetime(
            data_out["creation_ts"].dt.tz_localize("UTC")
        )
        data_out["creation_ts"] = pd.to_datetime(
            data_out["creation_ts"].dt.tz_convert(time_zone)
        )
    except TypeError:
        data_out["creation_ts"] = pd.to_datetime(
            data_out["creation_ts"].dt.tz_convert(time_zone)
        )
    try:
        data_out["validity_ts"] = pd.to_datetime(
            data_out["validity_ts"].dt.tz_localize("UTC")
        )
        data_out["validity_ts"] = pd.to_datetime(
            data_out["validity_ts"].dt.tz_convert(time_zone)
        )
    except TypeError:
        data_out["validity_ts"] = pd.to_datetime(
            data_out["validity_ts"].dt.tz_convert(time_zone)
        )
    message = f"Time columns have been converted to timezone {time_zone}."
    if verbose:
        print(message)

    data_out["step"] = data_out["validity_ts"] - data_out["creation_ts"]

    return data_out, nat_present


def outlier_detection_min_max(
    data: pd.DataFrame, min_value: float = -np.inf, max_value: float = np.inf
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outliers in the input data based on the min-max threshold values.

    Method: Min-max method to detect any data points outside the unbounded range
    (min_value, max_value) as outliers.

    Args:
        data: DataFrame with a datetime index.
        min_value: The minimum threshold value to detect outliers.
        max_value: The maximum threshold value to detect outliers.

    Returns:
        A tuple of two DataFrames with boolean values indicating outliers. The
        first DataFrame contains the lower outliers, and the second DataFrame
        contains the upper outliers.

    References:
        https://en.wikipedia.org/wiki/Min-max_scaling
    """
    outlier_mask_lower = data < min_value
    outlier_mask_upper = data > max_value
    return (outlier_mask_lower, outlier_mask_upper)


def outlier_detection_z_score(
    data: pd.DataFrame, threshold: float = 3.0
) -> tuple[pd.DataFrame]:
    """Detect outliers in the input data based on the z-score.

    Method: Z-score method to detect any data points with absolute z-scores
    greater than the given threshold as outliers.

    Args:
        data: DataFrame with a datetime index.
        threshold: The threshold value to detect outliers.

    Returns:
        A tuple with a DataFrame containing boolean values indicating outliers.

    References:
        https://en.wikipedia.org/wiki/Standard_score
    """
    z_scores = (data - data.mean()) / data.std()
    outlier_mask = z_scores.abs() > threshold
    return (outlier_mask,)


def outlier_detection_modified_z_score(
    data: pd.DataFrame, threshold: float = 3.5, verbose: bool = False
) -> tuple[pd.DataFrame]:
    """Detect outliers in the input data based on the modified z-score.

    Method: Modified z-score method to detect any data points with absolute
    modified z-scores greater than the given threshold as outliers.

    Args:
        data: DataFrame with a datetime index.
        threshold: The threshold value to detect outliers.
        verbose: A boolean flag to print additional information.

    Returns:
        A tuple with a DataFrame containing boolean values indicating outliers.

    References:
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

    """
    median = data.median()
    median_absolute_deviation = (data - median).abs().median()
    epsilon = 0.0
    if median_absolute_deviation == 0:
        epsilon = 1e-8
        message = (
            "Median absolute deviation is zero. Added a small tolerance "
            f"value of {epsilon} to avoid division by zero."
        )
        if verbose:
            print(message)
    modified_z_scores = 0.6745 * (data - median) / (median_absolute_deviation + epsilon)
    outlier_mask = modified_z_scores.abs() > threshold
    return (outlier_mask,)


def outlier_detection_iqr(
    data: pd.DataFrame, threshold: float = 1.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outliers in the input data based on the interquartile range.

    Method: Interquartile range (IQR) method to detect any data points outside
    the range defined by (Q1 - threshold * IQR, Q3 + threshold * IQR) as outliers.

    Args:
        data: DataFrame with a datetime index.
        threshold: The threshold value to detect outliers.

    Returns:
        A tuple of two DataFrames with boolean values indicating outliers. The
        first DataFrame contains the lower outliers, and the second DataFrame
        contains the upper outliers.

    References:
        https://en.wikipedia.org/wiki/Interquartile_range
        https://en.wikipedia.org/wiki/Robust_measures_of_scale
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    outlier_mask_lower = data < (q1 - threshold * iqr)
    outlier_mask_upper = data > (q3 + threshold * iqr)
    return (outlier_mask_lower, outlier_mask_upper)


def outlier_removal(
    data: pd.DataFrame,
    columns: list[str],
    bounds: tuple[float, float],
    method: str = "min_max",
    verbose: bool = False,
    **kwargs: float,
) -> pd.DataFrame:
    """Replace outliers in the input data based on the given values.

    Args:
        data: DataFrame with a datetime index.
        columns: List of column names to consider for outlier detection.
        bounds: A tuple of lower and upper bound values, index 0 and 1
            respectively, to replace outliers with. Both or one of lower bound
            or upper bound must be provided depeding on the outlier detection
            method used.
        method: The outlier detection method to use.
        verbose: A boolean flag to print additional information.
        **kwargs: Additional keyword arguments for the outlier detection method.

    Returns:
        A DataFrame with outliers replaced by the given values.

    Raises:
        ValueError:
            - If the outlier detection method is not supported.
            - If the bounds are not provided or are invalid.
            - If the bounds are not a tuple of length 2.
    """
    supported_outlier_detectors: dict[str, Callable[..., tuple[pd.DataFrame, ...]]] = {
        "z_score": outlier_detection_z_score,
        "modified_z_score": outlier_detection_modified_z_score,
        "iqr": outlier_detection_iqr,
        "min_max": outlier_detection_min_max,
    }

    if method not in supported_outlier_detectors:
        raise ValueError(
            f"Invalid outlier detection method: {method}. "
            f"Must be one of {supported_outlier_detectors.keys()}"
        )

    if isinstance(bounds, type(None)) or all(
        isinstance(bound, type(None)) for bound in bounds
    ):
        raise ValueError(
            "Argument bounds must be provided and cannot be a tuple of None."
        )
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError("Bounds must be a tuple of length 2.")

    outlier_mask = supported_outlier_detectors[method](data.loc[:, columns], **kwargs)
    if len(outlier_mask) == 1 and bounds[0] is None:
        bounds = (bounds[1], bounds[1])

    if verbose:
        _log_outliers(data, outlier_mask, bounds)

    for i, mask in enumerate(outlier_mask):
        if not isinstance(bounds[i], type(None)):
            data[mask] = bounds[i]
    return data


def _log_outliers(
    data: pd.DataFrame,
    outlier_mask: tuple[pd.DataFrame, ...],
    bounds: tuple[float, float],
) -> None:
    """Log useful information about the detected outliers.

    Args:
        data: DataFrame with a datetime index.
        outlier_mask: A tuple of DataFrames with boolean values indicating any
            detected outliers.
        bounds: A tuple of lower and upper bound values to replace outliers with.
    """
    log_messages = []
    log_messages += [
        f"Number of outliers found:\n{sum(mask.sum() for mask in outlier_mask)}"
    ]
    log_messages += [
        (
            "The following outlier values have been observed "
            f"and will be replaced with the value {bounds[i]}:\n"
            f"{[data.loc[mask[col], col].values for col in mask.columns]}\n"
            f"Outlier statistics: {data[mask.values].describe()}\n"
            if mask.values.any()
            else "No lower/upper outliers found."
        )
        for i, mask in enumerate(outlier_mask)
    ]
    for message in log_messages:
        print(message)
