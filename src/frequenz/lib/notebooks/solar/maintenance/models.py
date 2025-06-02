# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
This module contains the functions to prepare prediction models.

The main function `prepare_prediction_models` prepares prediction models based
on the provided specifications. The function takes the time series data with
timestamps as the index, a dictionary containing model labels as keys and their
corresponding specifications as values, and a list of model labels to extract
from the model specifications. It returns a dictionary with model labels as keys
and dictionaries containing the predictions as values. The predictions are stored
as pandas Series with the same index as the input data and the name 'predictions'.
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib
from numpy.typing import NDArray
from pandas import Index, Series
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import Array, FixedMount, PVSystem

ModelFunctionCallable = Callable[..., Any]
ModelFunction = ModelFunctionCallable | str
ColumnLabel = str | None
ResampleParams = tuple[str, str] | None
ModelParams = dict[str, Any]
ModelSpec = dict[str, ModelFunction | ColumnLabel | ResampleParams | ModelParams]

if TYPE_CHECKING:
    SeriesFloat = Series[float]
else:
    SeriesFloat = pd.Series  # Treated generically at runtime.


def prepare_prediction_models(
    data: pd.DataFrame, model_specs: dict[str, ModelSpec], keys_to_extract: list[str]
) -> dict[str, dict[str, SeriesFloat]]:
    """Prepare prediction models based on the provided specifications.

    Args:
        data: The time series data with timestamps as the index.
        model_specs: A dictionary containing model labels as keys and their
            corresponding specifications as values (a dictionary). Each
            specification dictionary contains:
                - 'model': A string or a callable representing the identifier
                    or the model function respectively. Currently, 4 models
                    are supported, with the following identifiers:
                    - 'wma' for weighted moving average.
                    - 'sampled_ma' for sampled moving average.
                    - 'naive_eff_irr2power' for naive efficiency factor model.
                    - 'pvlib' for running a simulation using PVLib.
                - 'target_label': An optional string representing the column
                    label to be used for processing. If None, the entire input
                    data will be used.
                - 'resample_params': An optional tuple containing resampling
                    parameters (rule, aggregation function) to be passed to
                    pd.Series.resample. If None, no resampling will be performed.
                - 'model_params': A dictionary containing parameters to be
                    passed to the model function.
        keys_to_extract: A list of model labels to extract from the model_specs.

    Returns:
        A dictionary with model labels as keys and dictionaries containing the
        predictions as values. The predictions are stored as pandas Series with
        the same index as the input data and the name 'predictions'.

    Raises:
        ValueError: If a specified model is not supported or if the target_label
            is not of type ColumnLabel or if the resample_params is not of type
            ResampleParams or if the model_params is not of type ModelParams.

    Example:
    # example set up of model_specs
    model_specs = {
        '7-day MA': {
            "model": "wma",
            'target_label': "energy_kWh",
            "resample_params": ("D", "sum"),
            "model_params": {"mode": "uniform", "win_size": 7}
        },
        '7-day MA - 15min': {
            "model": "sampled_ma",
            'target_label': "power_kW",
            "resample_params": ("15min", "sum"),
            "model_params": {"window": pd.Timedelta(days=7), "sampling_interval": 96}
        },
    }
    """
    data_copy = data.copy()

    supported_models: dict[str, Callable[..., SeriesFloat | NDArray[np.floating]]] = {
        "wma": weighted_moving_average,
        "sampled_ma": sampled_moving_average,
        "naive_eff_irr2power": naive_efficiency_factor_irr_to_power,
        "pvlib": run_pvlib_simulation,
    }

    prediction_models = {}
    for label, specs in (
        (key, model_specs[key]) for key in keys_to_extract if key in model_specs
    ):
        model_function = specs.get("model", "")
        target = specs.get("target_label", None)
        resample_params = specs.get("resample_params", None)
        model_params = specs.get("model_params", {})

        if not isinstance(target, ColumnLabel):
            raise ValueError(f"target_label must be of type {ColumnLabel}.")

        if not (
            isinstance(resample_params, type(None))
            or (
                isinstance(resample_params, tuple)
                and len(resample_params) == 2
                and all(isinstance(v, str) for v in resample_params)
            )
        ):
            raise ValueError(f"resample_params must be of type {ResampleParams}.")

        if not isinstance(model_params, dict):
            raise ValueError(f"model_params must be of type {ModelParams}.")

        if isinstance(model_function, str):
            if model_function not in supported_models:
                raise ValueError(f"Model '{model_function}' is not supported.")
            model_function = cast(
                ModelFunctionCallable, supported_models[model_function]
            )
        elif not callable(model_function):
            raise ValueError(f"Model '{model_function}' is not callable.")

        data_to_use = data_copy[[target]] if target else data_copy
        if resample_params:
            data_to_use = data_to_use.resample(resample_params[0]).agg(
                resample_params[1]
            )

        predictions = model_function(data=data_to_use, **model_params)

        if not isinstance(predictions, pd.Series):
            predictions = _align_predictions_to_index(predictions, data_to_use.index)

        prediction_models[label] = {"predictions": predictions}

    return prediction_models


def weighted_moving_average(
    data: NDArray[np.float64] | SeriesFloat | pd.DataFrame,
    *,
    mode: str | None = None,
    win_size: int | None = None,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.floating]:
    """Perform moving average with different weighting schemes.

    Args:
        data: The input data with datetime index.
        mode: Type of moving average with options:
            - 'uniform': The same weight is applied to all samples
            - 'recent': Uses weights of decreasing value for older samples
            - 'older': Uses weights of increasing value for older samples
            - 'exp': Uses exponential weights
        win_size: The size of the sliding window.
        weights: The weight values to apply on each sample, defined as [w_t,
            w_t-1, w_t-2, ...] i.e. the weight of most recent sample comes
            first. If defined, mode and win_size are ignored.

    Returns:
        An array of shape (len(data) - (win_size - 1),). The values correspond
        to weighted averages between past win_size - 1 samples and the current
        sample. In order to be used as predictions, one needs to shift the array
        accordingly so that it aligns with the original data and the prediction
        (i.e. the weighted average) corresponding to the ground truth does not
        include the ground truth itself.

    Raises:
        ValueError:
            - If the specified mode is not supported.
            - If the window size is not specified when weights are not given.
    """
    if weights is None:
        if win_size is None:
            raise ValueError(
                "Window size must be specified when weights are not given."
            )
        predefined_modes: dict[str, NDArray[np.float64]] = {
            "uniform": np.ones(win_size, dtype=np.float64),
            "recent": np.arange(1, win_size + 1, 1, dtype=np.float64)[::-1],
            "older": np.arange(1, win_size + 1, 1, dtype=np.float64),
            "exp": np.array(
                [(1 - 2 / (win_size + 1)) ** i for i in range(win_size)],
                dtype=np.float64,
            ),
        }
        if mode not in predefined_modes:
            raise ValueError(
                f"mode '{mode}' is not supported. Choose one of "
                f"{list(predefined_modes.keys())}."
            )
        weights = predefined_modes[mode]
    else:
        pass
    weights = weights / np.sum(weights)
    if isinstance(data, pd.DataFrame):
        return np.convolve(data.values[:, 0], weights, mode="valid")
    return np.convolve(data, weights, mode="valid")


def sampled_moving_average(
    data: SeriesFloat | pd.DataFrame,
    *,
    window: pd.Timedelta,
    sampling_interval: int,
    verbose: bool = False,
) -> SeriesFloat:
    """Compute the moving average of a time series.

    The moving average is computed over a specified window size, taking samples
    at specified step intervals within each moving window.

    Args:
        data: The input time series data with timestamps as the index.
        window: The moving window as a pandas Timedelta object.
        sampling_interval: The interval at which to sample the data within each
            window. Forced to be a positive integer.
        verbose: A boolean flag to print additional information.

    Returns:
        A pandas Series containing the moving averages, with the same index as
        the input data. Each value is the average of the samples taken at the
        specified interval within the moving window and includes the value at
        the index itself. In order to be used as predictions, one needs to shift
        the array accordingly so that it aligns with the original data and the
        prediction (i.e. the moving average) corresponding to the ground truth
        does not include the ground truth itself. This can be achieved by
        shifting the Series by sampling_interval.

    Raises:
        ValueError: If the input is a DataFrame with more than one column.
    """
    sampling_interval = abs(int(sampling_interval))
    data_index = pd.to_datetime(data.index).to_series()
    frequency = data_index.diff().median().total_seconds()
    if data_index.diff().std().total_seconds() > frequency * 0.1:
        message = (
            "The input series does not have a consistent frequency. "
            "Taking the most common frequency."
        )
        if verbose:
            print(message)
        frequency = data_index.diff().dt.total_seconds().mode().values[0]

    total_steps = int(window.total_seconds() / frequency)

    predictions = data.rolling(
        window=window, min_periods=total_steps, closed="left"
    ).apply(lambda x: x[::sampling_interval].mean(), raw=True)
    if isinstance(predictions, pd.DataFrame):
        if predictions.shape[1] != 1:
            raise ValueError("Expected predictions DataFrame to have one column.")
        predictions = predictions.iloc[:, 0]
    predictions.name = "predictions"
    return predictions


def naive_efficiency_factor_irr_to_power(  # pylint: disable=too-many-arguments
    data: pd.DataFrame,
    *,
    col_label: str,
    eff: float,
    peak_power_watts: float,
    rated_power_watts: float,
    resample_rate: str | pd.DateOffset | pd.Timedelta | None = None,
) -> SeriesFloat:
    """Compute the predicted power output based on the solar radiation data.

    Uses a naive approach in which one efficiency factor is used to model all
    inefficiencies in the system. The efficiency factor is applied to the solar
    radiation data and scaled by the peak power to compute the power output.
    The power output is then clipped to the rated power of the inverter(s).

    Args:
        data: The input time series data.
        col_label: The column label containing the solar radiation data.
        eff: The efficiency factor value.
        peak_power_watts: The total peak power of the solar system in watts.
        rated_power_watts: The total rated power of the inverters in watts.
        resample_rate: The rate at which to resample the data.

    Returns:
        A pandas Series containing the predicted power (in kilo-Watts) output.

    Raises:
        ValueError: If the input dataframe contains duplicate index entries or
            if the validity_ts is not unique.
    """
    data_copy = data.copy(deep=True)
    data_copy.sort_values(by=["creation_ts", "validity_ts"], inplace=True)
    data_copy[col_label] = data_copy[col_label].fillna(0).clip(lower=0)

    to_include = (data_copy["step"] >= timedelta(hours=1)) & (
        data_copy["step"] <= timedelta(hours=6)
    )
    position = to_include.index.get_loc(to_include[::-1].idxmax())
    if not isinstance(position, int):
        raise ValueError("Input dataframe contains duplicate index entries.")
    if position + 1 < len(to_include):
        to_include.loc[to_include.index[position + 1 :]] = True
    data_copy = data_copy[to_include]

    if data_copy.validity_ts.nunique() != data_copy.validity_ts.count():
        raise ValueError("validity_ts is not unique")
    data_copy.set_index("validity_ts", inplace=True)
    data_copy.index.name = "timestamp"
    if resample_rate:
        ssr = data_copy[col_label].resample(resample_rate).interpolate()
    else:
        ssr = data_copy[col_label]
    predictions = (eff * ssr * peak_power_watts / 1000).clip(
        upper=rated_power_watts
    ) / 1000
    predictions.name = "predictions"
    return predictions


# pylint: disable-next=too-many-arguments, too-many-locals
def run_pvlib_simulation(  # pylint: disable=unused-argument
    data: pd.DataFrame,
    *,
    location_parameters: dict[str, Any],
    pv_system_arrays: list[dict[str, Any]],
    inverter_parameters: dict[str, Any],
    start_year: int,
    end_year: int,
    sampling_rate: str | pd.DateOffset | pd.Timedelta | None = None,
    weather_option: str = "tmy",
    time_zone: ZoneInfo = ZoneInfo("UTC"),
) -> SeriesFloat:
    """Run a PVLib simulation using the provided parameters for a single site.

    Note: This is WIP. The PV system set up needs more work to become more general
    and flexible. The parameters should eventually be exposed to the user. For
    now they are fixed to east and west arrays so that simulation works as a demo.

    Args:
        data: Placeholder for the input data. Not used in this function.
        location_parameters: A dictionary containing the location parameters
            with the following keys:
            - 'latitude': The latitude of the location.
            - 'longitude': The longitude of the location.
            - 'altitude': The altitude of the location.
            - 'timezone': The timezone of the location.
            - 'name': The name of the location.
        pv_system_arrays: A list of dictionaries, each containing the following
            keys:
            - 'name': The name of the array.
            - 'surface_tilt': The tilt angle of the surface in degrees.
            - 'surface_azimuth': The azimuth angle of the surface in degrees.
            - 'module': A dictionary containing the module parameters.
            - 'modules_per_string': The number of modules per string.
            - 'temperature_parameters': A dictionary containing the temperature
                model parameters.
            - 'strings': The number of strings.
        inverter_parameters: A dictionary containing the inverter parameters.
        start_year: The start year for the simulation.
        end_year: The end year for the simulation.
        sampling_rate: The rate at which to resample the data.
        weather_option: The weather data option to use. Choose one of:
            - 'tmy': Typical Meteorological Year. Considers at least 10 years in
                identifying the most typical month for each month.
            - 'hourly': Get hourly historical solar irradiation and modeled PV
                power output from PVGIS by specifying the start and end years
                from which to get the data.
        time_zone: The timezone to convert the index of the returned Series to.
            Should be a valid zoneinfo.ZoneInfo object. Defaults to 'UTC'.

    Returns:
        A pandas Series containing the power predictions in Watts.

    Raises:
        ValueError: If the specified weather option is not supported.
    """

    def _get_pvgis_tmy(
        latitude: float,
        longitude: float,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """Get TMY data from PVGIS database.

        Args:
            latitude: The latitude of the location.
            longitude: The longitude of the location.
            start_year: The start year for the TMY data. The year must be
                between 2005 and 2020.
            end_year: The end year for the TMY data. The year must be
                between 2005 and 2020.

        Returns:
            A pandas DataFrame containing the Typical Meteorological Year (TMY)
            data.

        References:
        - https://pvlib-python.readthedocs.io/en/stable/reference/generated/
        pvlib.iotools.get_pvgis_tmy.html
        """
        tmy_df, _, _, _ = pvlib.iotools.get_pvgis_tmy(
            latitude=latitude,
            longitude=longitude,
            startyear=start_year,
            endyear=end_year,
            url="https://re.jrc.ec.europa.eu/api/v5_2/",
        )
        tmy_df.index = tmy_df.index.map(lambda x: x.replace(year=datetime.now().year))
        tmy_df.sort_index(inplace=True)
        tmy_df.index = tmy_df.index.tz_convert(location_parameters["timezone"])

        tmy_weather: pd.DataFrame = tmy_df.loc[:, ["ghi", "dni", "dhi"]]
        return tmy_weather

    def _get_pvgis_hourly(
        latitude: float,
        longitude: float,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """Get hourly data from PVGIS database.

        Args:
            latitude: The latitude of the location.
            longitude: The longitude of the location.
            start_year: The start year of the radiation time series. The year
                must be between 2005 and 2020.
            end_year: The end year of the radiation time series. The year
                must be between 2005 and 2020.

        Returns:
            A pandas DataFrame containing the hourly data.

        Raises:
            ValueError: If the minute values in the index are not unique.

        References:
        - https://pvlib-python.readthedocs.io/en/stable/reference/generated/
        pvlib.iotools.get_pvgis_hourly.html
        """
        pvgis_df, inputs, metadata = pvlib.iotools.get_pvgis_hourly(
            latitude=latitude,
            longitude=longitude,
            start=start_year,
            end=end_year,
            raddatabase="PVGIS-SARAH2",
            url="https://re.jrc.ec.europa.eu/api/v5_2/",
        )
        if pvgis_df.index.minute.nunique() != 1:
            raise ValueError("Minute values in the index are not unique.")
        pvgis_df.index = pvgis_df.index - pd.Timedelta(
            minutes=pvgis_df.index.minute.unique()[0]
        )
        year_offset = datetime.now().year - end_year
        pvgis_df.index = pvgis_df.index.map(
            lambda x: x.replace(year=x.year + year_offset)
        )
        pvgis_df.index = pvgis_df.index.tz_convert(location_parameters["timezone"])

        dhi_pvgis = pvgis_df["poa_sky_diffuse"] + pvgis_df["poa_ground_diffuse"]
        dni_pvgis = pvgis_df["poa_direct"]
        ghi_pvgis = dni_pvgis + dhi_pvgis

        pvgis_weather: pd.DataFrame = pd.concat(
            [ghi_pvgis.rename("ghi"), dni_pvgis.rename("dni"), dhi_pvgis.rename("dhi")],
            axis=1,
            ignore_index=False,
        )
        return pvgis_weather

    valid_weather_options = {"tmy": _get_pvgis_tmy, "hourly": _get_pvgis_hourly}
    weather_data_function = valid_weather_options.get(weather_option, None)
    if weather_data_function is None:
        raise ValueError(
            f"Invalid weather option '{weather_option}'. Choose one of "
            f"{list(valid_weather_options.keys())}."
        )
    weather_data = weather_data_function(
        location_parameters["latitude"],
        location_parameters["longitude"],
        start_year,
        end_year,
    )

    # --- Set up the PV system --- #
    location = Location(
        latitude=location_parameters["latitude"],
        longitude=location_parameters["longitude"],
        altitude=location_parameters["altitude"],
        tz=location_parameters["timezone"],
        name=location_parameters["name"],
    )

    arrays = [
        Array(
            FixedMount(
                surface_tilt=array["surface_tilt"],
                surface_azimuth=array["surface_azimuth"],
            ),
            name=array["name"],
            module_parameters=array["module"],
            modules_per_string=array["modules_per_string"],
            temperature_model_parameters=array["temperature_parameters"],
            strings=array["strings"],
        )
        for array in pv_system_arrays
    ]

    system = PVSystem(
        name=location_parameters["name"],
        arrays=arrays,
        inverter_parameters=inverter_parameters,
    )
    # ---------------------------- #

    modelchain = ModelChain(system=system, location=location)
    modelchain.run_model(weather=weather_data)

    simulation_data = pd.DataFrame(modelchain.results.ac)
    simulation_data["p_mp"] = simulation_data["p_mp"].fillna(0).clip(lower=0)
    simulation_data.rename(columns={"p_mp": "power_W"}, inplace=True)
    if sampling_rate:
        simulation_data = simulation_data.resample(sampling_rate).interpolate()
    simulation_data.index.name = "timestamp"
    simulation_data_index = pd.to_datetime(simulation_data.index)
    try:
        simulation_data.index = pd.to_datetime(
            simulation_data_index.tz_localize(time_zone)
        )
    except TypeError:
        simulation_data.index = pd.to_datetime(
            simulation_data_index.tz_convert(time_zone)
        )

    predictions: SeriesFloat = simulation_data["power_W"]
    predictions.name = "predictions"
    return predictions


def _align_predictions_to_index(
    predictions: NDArray[np.float64],
    reference_index: "Index[Any]",
) -> SeriesFloat:
    """Align predictions to a reference index, padding with NaNs if necessary.

    Args:
        predictions: The prediction outputs.
        reference_index: The DataFrame index to align to.

    Returns:
        A pandas Series with predictions aligned to the reference index.

    Note:
        If predictions are longer than the reference index, predictions will
        be truncated from the left. If predictions are shorter, they are
        right-aligned and earlier entries are filled with NaN.
    """
    reference_length = len(reference_index)
    prediction_length = len(predictions)

    if prediction_length > reference_length:
        predictions = predictions[-reference_length:]
    elif prediction_length < reference_length:
        padded_predictions = np.full(reference_length, np.nan, dtype=np.float64)
        padded_predictions[-prediction_length:] = predictions
        predictions = padded_predictions

    return pd.Series(data=predictions, index=reference_index, name="predictions")
