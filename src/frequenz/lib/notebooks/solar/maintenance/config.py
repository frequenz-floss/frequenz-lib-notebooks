# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
This module contains the SolarMaintenanceConfig class.

The SolarMaintenanceConfig class is a configuration class that contains
parameters and methods to update them dynamically, ensuring consistency and
preventing unauthorized modifications. The class inherits from ConfigConstants,
which contains immutable constants for the Solar Maintenance Project.
"""

import datetime
import warnings
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from frequenz.client.common.metric import Metric


class ConfigConstants:
    """Class containing immutable constants for the Solar Maintenance Project.

    Constants:
        ATTRIBUTES_THAT_CANNOT_BE_UPDATED (list[str]): Attributes that cannot
            be updated directly.
        VALID_OUTLIER_DETECTION_METHODS (list[str]): Supported outlier
            detection methods. Currently supported methods are:
                - 'z_score'
                - 'modified_z_score'
                - 'min_max'
                - 'iqr'
        VALID_ROLLING_VIEW_TIME_FRAMES (list[str]): Supported time frames for
            the rolling view plot. Currently only 'days' is supported.
        SUPPORTED_LANGUAGES (list[str]): Supported languages for translations.
            Currently supported languages are: 'en' (English) and 'de' (German).
        RANGE_SMALL_RESAMPLE_PERIOD (tuple[int, int]): Bounded range (min, max)
            in seconds, for the small_resample_period_seconds
        RANGE_REAL_TIME_VIEW_DURATION_HOURS (tuple[int, int]): Bounded range
            (min, max) for the real time view duration in hours.
        VALID_STAT_PROFILE_GROUPINGS (list[str]): Supported groupings for the
            statistical profile. Currently supported groupings are:
                - 'grouped': Groups data by an interval size defined by
                        DEFAULT_LARGE_RESAMPLE_PERIOD_SECONDS in minutes, merging
                        data from the same interval across different days into
                        one group.
                - 'continuous': Groups data into continuous intervals throughout
                        the entire dataset duration, without merging same-interval
                        data across different days. The interval size is defined
                        by DEFAULT_LARGE_RESAMPLE_PERIOD_SECONDS in minutes.
                - '24h_continuous': Groups data into continuous 24-hour intervals
                        throughout the entire dataset duration, without merging
                        same-interval data across different days.
        VALID_BASELINE_MODELS (list[str]): Supported baseline models. Currently
            supported models are:
                - '7-day MA': 7-day moving average.
                - '7-day sampled MA': 7-day moving average with intervals of
                    DEFAULT_LARGE_RESAMPLE_PERIOD_SECONDS in minutes.
                - 'simulation': Simulation-based forecast using pvlib library.
                - 'weather-based-forecast': Weather-based forecast using data
                    from the weather API.
        VALID_PLOT_THEMES (list[str]): Supported plot themes. Currently
            supported themes are:
                - 'frequenz-neustrom' (default)
                - 'elegant-minimalist'
                - 'vibrant'
                - 'classic'
        DEFAULT_VERBOSITY_FLAG (bool): Default verbosity flag. A boolean flag
            to print additional information to the console.
        DEFAULT_WEATHER_SERVICE_ADDRESS (str): Default weather service address.
        DEFAULT_REPORTING_SERVICE_ADDRESS (str): Default reporting service
            address.
        DEFAULT_MICROGRID_IDS (list[int]): Default microgrid IDs.
        DEFAULT_COMPONENT_IDS (list[list[int]]): Default component IDs for each
            microgrid.
        DEFAULT_OUTLIER_DETECTION_PARAMS (dict): Default parameters for outlier
            detection. Contains these keys:
                - method: Outlier detection method.
                - bounds: The lower and upper values, at index 0 and 1 respectively,
                    to replace outliers with.
                - params: Additional parameters for the outlier detection method.
        DEFAULT_METRICS_TO_FETCH (list[Metric]): Default metrics to fetch from
            the reporting API.
        DEFAULT_SMALL_RESAMPLE_PERIOD_SECONDS (int): Default small sample period.
        DEFAULT_LARGE_RESAMPLE_PERIOD_SECONDS (int): Default large sample period.
        DEFAULT_WEATHER_FEATURES_TO_FETCH (list[str]): Default features to fetch
            from the weather API.
        DEFAULT_WEATHER_FEATURE_NAMES (list[str]): Default internal feature names
            for the weather API features. These must match the order of
            DEFAULT_WEATHER_FEATURES_TO_FETCH.
        DEFAULT_TIME_ZONE (ZoneInfo): Timezone for all timestamps. Default is UTC.
        DEFAULT_START_TIMESTAMP (datetime): Default start timestamp for fetching
            data.
        DEFAULT_END_TIMESTAMP (datetime): Default end timestamp for fetching
            data.
        DEFAULT_CLIENT_SITE_INFO_KEYS_VALUE_TYPES (list[dict[str, tuple[type, Any]]]):
            List of one dictionary that contains the default keys and value
            types (along with dummy default values) for the client site info.
            Contains these keys:
            - name (str): Name of the client site.
            - latitude (float): The latitude of the location.
            - longitude (float): The longitude of the location.
            - altitude (float): The altitude of the location.
            - peak_power_watts (float): The peak power (solar panels) in Watts.
            - rated_power_watts (float): The rated power (PV inverters) in Watts.
            - efficiency (float): The overall system efficiency. Range: [0.0, 1.0].
            - mid (int): The microgrid ID of the client site.
            - malo_id (str): The MALO ID of the client site.
        DEFAULT_ROLLING_VIEW_DURATION (int): Default duration for the rolling
            view plot.
        DEFAULT_ROLLING_VIEW_TIME_FRAME (str): Default time frame for the
            rolling view plot.
        DEFAULT_REAL_TIME_VIEW_DURATION_HOURS (int): Default duration for the
            real time view in hours.
        DEFAULT_STAT_PROFILE_GROUPING (list[str]): Default groupings for the
            statistical profile.
        DEFAULT_PLOT_THEME (str): Default plot theme for the plots.
        DEFAULT_FORCE_POSITIVE_VALUES (bool): Default flag to force positive
            values for all data points.
        DEFAULT_SPLIT_REAL_TIME_VIEW_PER_INVERTER (bool): Default flag to split
            the real time view per inverter (True) or to show the total power
            (False).
        DEFAULT_BASELINE_MODELS (list[str]): Default baseline models to use.
        DEFAULT_LANGUAGE (str): Default language for the translations.
    """

    ATTRIBUTES_THAT_CANNOT_BE_UPDATED = [
        "microgrid_components"
    ]  # NOTE: any config argument that cannot be updated should not be a config argument
    VALID_OUTLIER_DETECTION_METHODS = ["z_score", "modified_z_score", "min_max", "iqr"]
    VALID_ROLLING_VIEW_TIME_FRAMES = ["days"]
    SUPPORTED_LANGUAGES = ["en", "de"]
    RANGE_SMALL_RESAMPLE_PERIOD = (1, 10)
    RANGE_REAL_TIME_VIEW_DURATION_HOURS = (0.25, 3)
    VALID_STAT_PROFILE_GROUPINGS = [
        "grouped",
        "continuous",
        "24h_continuous",
    ]
    VALID_BASELINE_MODELS = [
        "7-day MA",
        "7-day sampled MA",
        "simulation",
        "weather-based-forecast",
    ]
    VALID_PLOT_THEMES = [
        "frequenz-neustrom",
        "elegant-minimalist",
        "vibrant",
        "classic",
    ]
    DEFAULT_VERBOSITY_FLAG = False
    DEFAULT_WEATHER_SERVICE_ADDRESS = "http://weatherapi.example.com"
    DEFAULT_REPORTING_SERVICE_ADDRESS = "http://reportingapi.example.com"
    DEFAULT_MICROGRID_IDS = [0]
    DEFAULT_COMPONENT_IDS = [[0]]
    DEFAULT_OUTLIER_DETECTION_PARAMS: dict[
        str, str | tuple[float, float] | dict[str, Any]
    ] = {
        "method": "min_max",
        "bounds": (-np.inf, 0.0),
        "params": {"min_value": -np.inf, "max_value": 0.0},
    }
    DEFAULT_METRICS_TO_FETCH = [Metric.AC_ACTIVE_POWER]
    DEFAULT_LARGE_RESAMPLE_PERIOD_SECONDS = 60 * 15
    DEFAULT_WEATHER_FEATURES_TO_FETCH = ["SURFACE_NET_SOLAR_RADIATION"]
    DEFAULT_WEATHER_FEATURE_NAMES = ["ssr"]
    DEFAULT_TIME_ZONE = ZoneInfo("UTC")
    DEFAULT_END_TIMESTAMP = datetime.datetime.now().astimezone(
        DEFAULT_TIME_ZONE
    ) - datetime.timedelta(minutes=15)
    DEFAULT_START_TIMESTAMP = DEFAULT_END_TIMESTAMP - datetime.timedelta(days=1)
    DEFAULT_CLIENT_SITE_INFO_KEYS_VALUE_TYPES = [
        {
            "name": (str, "client"),
            "latitude": (float, 0.0),
            "longitude": (float, 0.0),
            "altitude": (float, 0.0),
            "peak_power_watts": (float, 0.0),
            "rated_power_watts": (float, 0.0),
            "efficiency": (float, 1.0),
            "mid": (int, -1),
            "malo_id": (str, ""),
        }
    ]
    DEFAULT_ROLLING_VIEW_DURATION = 30
    DEFAULT_ROLLING_VIEW_TIME_FRAME = "days"
    DEFAULT_SMALL_RESAMPLE_PERIOD_SECONDS = 2
    DEFAULT_REAL_TIME_VIEW_DURATION_HOURS = 2
    DEFAULT_STAT_PROFILE_GROUPING = [
        "grouped",
        "continuous",
        "24h_continuous",
    ]
    DEFAULT_PLOT_THEME = "frequenz-neustrom"
    DEFAULT_FORCE_POSITIVE_VALUES = True
    DEFAULT_SPLIT_REAL_TIME_VIEW_PER_INVERTER = True
    DEFAULT_BASELINE_MODELS = [
        "7-day MA",
        "7-day sampled MA",
        "weather-based-forecast",
    ]
    DEFAULT_LANGUAGE = "en"


# pylint: disable=too-many-instance-attributes
class SolarMaintenanceConfig(ConfigConstants):
    """Configuration class the solar maintenance project.

    This class contains parameters and methods to update them dynamically,
    ensuring consistency and preventing unauthorized modifications.

    Attributes:
        weather_service_address (str): Endpoint for weather data API.
        reporting_service_address (str): Endpoint for reporting data API.
        weather_feature_names_mapping (dict): Mapping of weather API feature
            names to internal feature names.
        microgrid_ids (list[int]): List of microgrid IDs. Each microgrid ID must
            correspond to the element of the same index in component_ids.
        component_ids (list[list[int]]): List of component IDs for each microgrid.
        microgrid_components (list[tuple[int, list[int]]]): List of microgrid-
            component ID pairs.
        metrics_to_fetch (list[Metric]): List of metrics to fetch from the
            reporting API.
        small_resample_period_seconds (int): The sample period value in seconds
            to use when fetching data from the reporting API for further processing
            and/or plotting that requires high temporal resolution.
        large_resample_period_seconds (int): The sample period value in seconds
            to use when fetching data from the reporting API for further processing
            and/or plotting that does not require high temporal resolution.
        time_zone (ZoneInfo): Timezone for the data. Default is UTC. To set a
            different timezone, the set_timezone method is used which accepts
            a string in the IANA timezone database format.
        start_timestamp (datetime): Start timestamp for fetching data (in UTC).
        end_timestamp (datetime): End timestamp for fetching data (in UTC).
        client_site_info (list[dict[str, Any]]): Information per client site.
            Each List element must correspond to the same index in microgrid_ids
            and component_ids. Each dictionary can be empty (in which case all
            the default values are kept) or contain one or more of the keys in
            DEFAULT_CLIENT_SITE_INFO_KEYS_VALUE_TYPES (in which case the default
            values are overwritten).
        outlier_detection_parameters (dict | None): Parameters for outlier
            detection. See DEFAULT_OUTLIER_DETECTION_PARAMS for more details
            and see VALID_OUTLIER_DETECTION_METHODS for the supported methods.
        rolling_view_duration (int): Duration for the rolling view plot.
        rolling_view_time_frame (str): Time frame for the rolling view plot.
        stat_profile_grouping (list[str]): Groupings for the statistical profile.
            See VALID_STAT_PROFILE_GROUPINGS for the supported groupings.
        verbose (bool): Verbosity flag. A boolean flag to print additional info.
        real_time_view_duration_hours (int): Duration for the real time view in
            hours.
        plot_theme (str): Theme for the plots. See VALID_PLOT_THEMES for the
            supported themes.
        force_positive_values (bool): Flag to force positive values for all
            data points.
        split_real_time_view_per_inverter (bool): Flag to split the real time
            view per inverter (True) or to show the total power (False).
        baseline_models (list[str]): Baseline models to use. See
            VALID_BASELINE_MODELS for the supported models.
        language (str): Language for the translations.

    Constants:
        Inherited from ConfigConstants. These include:
        - VALID_OUTLIER_DETECTION_METHODS: List of valid methods for outlier
            detection.
        - DEFAULT_*: Default values for various configuration parameters.

    Methods:
        set_default_parameters: Set default parameters for the configuration.
        update_parameter: Update a specific parameter with a new value.
        update_mids_and_cids: Update microgrid_ids and component_ids together.
        update_dict: Update specific values in a dictionary.
        validate_all_settings: Validate all configuration settings.

    Notes:
        - The configuration parameters are set to default values in the
            constructor.
        - The default service addresses, microgrid IDs, component IDs and
            client_site_info are set to example values for demonstration
            purposes. These should be updated with the actual values before
            running the application.
    """

    def __init__(self) -> None:
        """Initialize the SolarMaintenanceConfig class."""
        # --- weather service parameters --- #
        self.weather_service_address = self.DEFAULT_WEATHER_SERVICE_ADDRESS
        self.weather_feature_names_mapping = dict(
            zip(
                self.DEFAULT_WEATHER_FEATURES_TO_FETCH,
                self.DEFAULT_WEATHER_FEATURE_NAMES,
            )
        )
        # ---------------------------------- #

        # --- reporting service parameters --- #
        self.reporting_service_address = self.DEFAULT_REPORTING_SERVICE_ADDRESS
        self.microgrid_ids: list[int] = self.DEFAULT_MICROGRID_IDS
        self.component_ids: list[list[int]] = self.DEFAULT_COMPONENT_IDS
        self._update_microgrid_components()
        self.metrics_to_fetch = self.DEFAULT_METRICS_TO_FETCH
        self.large_resample_period_seconds = self.DEFAULT_LARGE_RESAMPLE_PERIOD_SECONDS
        # ---------------------------------- #

        # --- other parameters --- #
        self.end_timestamp = self.DEFAULT_END_TIMESTAMP
        self.start_timestamp = self.DEFAULT_START_TIMESTAMP
        self.time_zone = self.DEFAULT_TIME_ZONE
        self.client_site_info = [
            {key: value[1] for key, value in d.items()}
            for d in self.DEFAULT_CLIENT_SITE_INFO_KEYS_VALUE_TYPES
        ]
        self.outlier_detection_parameters = self.DEFAULT_OUTLIER_DETECTION_PARAMS
        self.rolling_view_duration = self.DEFAULT_ROLLING_VIEW_DURATION
        self.rolling_view_time_frame = self.DEFAULT_ROLLING_VIEW_TIME_FRAME
        self.stat_profile_grouping = self.DEFAULT_STAT_PROFILE_GROUPING
        self.verbose = self.DEFAULT_VERBOSITY_FLAG
        self.small_resample_period_seconds = self.DEFAULT_SMALL_RESAMPLE_PERIOD_SECONDS
        self.real_time_view_duration_hours = self.DEFAULT_REAL_TIME_VIEW_DURATION_HOURS
        self.plot_theme = self.DEFAULT_PLOT_THEME
        self.force_positive_values = self.DEFAULT_FORCE_POSITIVE_VALUES
        self.split_real_time_view_per_inverter = (
            self.DEFAULT_SPLIT_REAL_TIME_VIEW_PER_INVERTER
        )
        self.baseline_models = self.DEFAULT_BASELINE_MODELS
        self.language = self.DEFAULT_LANGUAGE
        # ---------------------------------- #
        self.validate_all_settings()

    def update_parameter(self, param_name: str, param_value: Any) -> None:
        """Update parameters dynamically with type and logical checks.

        Note: Certain parameters cannot be directly updated and require specific
              methods. For example, microgrid_ids and component_ids should be
              updated using update_mids_and_cids.

        Args:
            param_name: The name of the parameter to update.
            param_value: The new value for the parameter.

        See Also:
            `validate_all_settings()` and `_validate_can_be_updated()` for
            validation details.
        """
        self._validate_can_be_updated(
            param_name,
            self.ATTRIBUTES_THAT_CANNOT_BE_UPDATED + ["microgrid_ids", "component_ids"],
            "Use update_mids_and_cids updates for microgrid_ids and component_ids.",
        )
        if hasattr(self, param_name):
            current_value = getattr(self, param_name)
            if isinstance(current_value, dict):
                self.update_dict(current_value, param_value, param_name)
            else:
                if param_name == "time_zone":
                    self.set_time_zone(param_value)
                elif param_name in ["start_timestamp", "end_timestamp"]:
                    self.set_timestamp(param_name, param_value)
                else:
                    setattr(self, param_name, param_value)
            message = f"Parameter '{param_name}' updated to '{param_value}'."
            if self.verbose:
                print(message)
        else:
            message = (
                f"WARNING: Parameter '{param_name}' not found in "
                "SolarMaintenanceConfig and is not added."
            )
            if self.verbose:
                warnings.warn(message)
        self.validate_all_settings()

    def update_mids_and_cids(
        self, microgrid_ids: list[int], component_ids: list[list[int]]
    ) -> None:
        """Update microgrid_ids and component_ids together.

        Args:
            microgrid_ids: A list of microgrid IDs.
            component_ids: A list of lists of component IDs.

        See Also:
            see validate_all_settings() for validation details.
        """
        self.microgrid_ids = microgrid_ids
        self.component_ids = component_ids
        self._update_microgrid_components()
        message = (
            f"Updated microgrid IDs to '{microgrid_ids}' "
            f"and component IDs to '{component_ids}'."
        )
        if self.verbose:
            print(message)
        self.validate_all_settings()

    def update_dict(
        self, dict_to_update: dict[str, Any], updates: dict[str, Any], param_name: str
    ) -> None:
        """Update specific key values in a dictionary.

        Args:
            dict_to_update: Dictionary to update.
            updates: Dictionary containing the values to update.
            param_name: The name of the parameter being updated.

        See Also:
            `_validate_inclusion()` and `validate_all_settings()` for validation
            details.
        """
        if param_name == "outlier_detection_parameters":
            self._validate_type(updates, (dict, type(None)), "updates")
            if not updates:
                setattr(self, param_name, updates)
                return
        else:
            self._validate_type(updates, dict, "updates")
        for key, value in updates.items():
            if key not in dict_to_update:
                self._validate_inclusion(key, list(dict_to_update.keys()), "key")
            dict_to_update[key] = value
            message = f"Updated '{key}' to '{value}'."
            if self.verbose:
                print(message)
        self.validate_all_settings()

    def set_time_zone(self, time_zone_str: str) -> None:
        """Set the timezone for the data.

        Args:
            time_zone_str: The timezone to set. Should be in the IANA timezone
                database format (i.e. supported by zoneinfo).

        Raises:
            ValueError: If the timezone is not found in the zoneinfo database.
        """
        try:
            self.time_zone = ZoneInfo(time_zone_str)
            # Update timestamps to new timezone
            self.set_timestamp("start_timestamp", self.start_timestamp)
            self.set_timestamp("end_timestamp", self.end_timestamp)
            message = (
                f"Parameters 'start_timestamp': {self.start_timestamp} and "
                f"'end_timestamp': {self.end_timestamp} updated to the new timezone."
            )
            if self.verbose:
                print(message)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"Invalid timezone: '{time_zone_str}'. "
                "Please provide a valid IANA timezone name."
            ) from exc

    def set_timestamp(self, param_name: str, timestamp: datetime.datetime) -> None:
        """Set the start or end timestamp for fetching data.

        Args:
            param_name: The name of the parameter to update.
            timestamp: The new timestamp to set.

        Raises:
            ValueError: If the timestamp is not a datetime object.
        """
        if not isinstance(timestamp, datetime.datetime):
            raise ValueError("Timestamp must be a datetime object.")
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.time_zone)
        else:
            timestamp = timestamp.astimezone(self.time_zone)
        if param_name == "start_timestamp":
            self.start_timestamp = timestamp
        elif param_name == "end_timestamp":
            self.end_timestamp = timestamp
        self.validate_all_settings()

    def _update_microgrid_components(self) -> None:
        """Update microgrid_components based on microgrid_ids and component_ids.

        See Also:
            see validate_all_settings() for more details.
        """
        try:
            self.microgrid_components: list[tuple[int, list[int]]] = list(
                zip(self.microgrid_ids, self.component_ids)
            )
        except (TypeError, ValueError, AttributeError):
            self.validate_all_settings()

    def validate_all_settings(self) -> None:
        """Validate all configuration settings.

        See Also:
            _validate_service_address(), _validate_type(),
            _validate_value_range(), _validate_list_of(),
            _validate_matching_lengths(), _validate_dictionary(),
            _validate_inclusion(), _validate_can_be_updated() for validation
            details.
        """
        self._validate_service_address(self.weather_service_address, "weather service")
        self._validate_service_address(
            self.reporting_service_address, "reporting service"
        )
        self._validate_type(
            self.weather_feature_names_mapping, dict, "weather feature names mapping"
        )
        self._validate_type(self.metrics_to_fetch, list, "metrics to fetch")
        self._validate_type(
            self.large_resample_period_seconds, int, "large resample period seconds"
        )
        self._validate_type(self.start_timestamp, datetime.datetime, "start timestamp")
        self._validate_type(self.end_timestamp, datetime.datetime, "end timestamp")
        self._validate_type(self.client_site_info, list, "client site info")
        self._validate_type(
            self.outlier_detection_parameters,
            (dict, type(None)),
            "outlier detection parameters",
        )
        self._validate_type(self.verbose, bool, "verbosity flag")
        self._validate_type(self.rolling_view_duration, int, "rolling view duration")
        self._validate_type(
            self.rolling_view_time_frame, str, "rolling view time frame"
        )
        self._validate_type(
            self.small_resample_period_seconds, int, "small resample period seconds"
        )
        self._validate_type(
            self.real_time_view_duration_hours, int, "real time view duration"
        )
        self._validate_type(self.plot_theme, str, "plot theme")
        self._validate_type(self.force_positive_values, bool, "force positive values")
        self._validate_type(
            self.split_real_time_view_per_inverter,
            bool,
            "split real time view per inverter",
        )
        self._validate_value_range(
            self.small_resample_period_seconds,
            self.RANGE_SMALL_RESAMPLE_PERIOD[0],
            self.RANGE_SMALL_RESAMPLE_PERIOD[1],
            "small resample period seconds",
        )
        self._validate_value_range(
            self.real_time_view_duration_hours,
            self.RANGE_REAL_TIME_VIEW_DURATION_HOURS[0],
            self.RANGE_REAL_TIME_VIEW_DURATION_HOURS[1],
            "real time view duration",
        )
        self._validate_type(self.stat_profile_grouping, list, "stat profile grouping")
        self._validate_type(self.baseline_models, list, "baseline models")
        self._validate_type(self.microgrid_ids, list, "microgrid IDs")
        self._validate_type(self.component_ids, list, "component IDs")
        self._validate_type(self.microgrid_components, list, "microgrid components")
        self._validate_type(self.language, str, "language")
        self._validate_list_of(self.microgrid_ids, int, "microgrid IDs")
        self._validate_list_of(self.component_ids, list, "component IDs")
        self._validate_list_of(self.microgrid_components, tuple, "microgrid components")
        self._validate_matching_lengths(
            self.microgrid_ids, self.component_ids, "microgrid IDs", "component IDs"
        )
        if self.outlier_detection_parameters:
            self._validate_inclusion(
                self.outlier_detection_parameters["method"],
                self.VALID_OUTLIER_DETECTION_METHODS,
                "outlier detection method",
            )
        self._validate_inclusion(
            self.rolling_view_time_frame,
            self.VALID_ROLLING_VIEW_TIME_FRAMES,
            "rolling view time frame",
        )
        self._validate_inclusion(
            self.language,
            self.SUPPORTED_LANGUAGES,
            "language",
        )
        for grouping in self.stat_profile_grouping:
            self._validate_inclusion(
                grouping,
                self.VALID_STAT_PROFILE_GROUPINGS,
                "stat profile grouping",
            )
        for model in self.baseline_models:
            self._validate_inclusion(
                model,
                self.VALID_BASELINE_MODELS,
                "baseline model",
            )
        self._validate_inclusion(
            self.plot_theme,
            self.VALID_PLOT_THEMES,
            "plot theme",
        )
        for d in self.client_site_info:
            self._validate_dictionary(
                d,
                self.DEFAULT_CLIENT_SITE_INFO_KEYS_VALUE_TYPES[0],
                "Client Site Info",
            )

    def _validate_service_address(self, address: str, name: str) -> None:
        """Validate the format of a service address.

        Args:
            address: The service address to validate.
            name: The name of the service.

        Raises:
            ValueError: If the address is not a valid URL.
        """
        if not address.startswith(("fz", "http", "https", "reporting", "grpc")):
            raise ValueError(f"Invalid {name} address: {address}")

    def _validate_type(self, variable: Any, expected_type: Any, name: str) -> None:
        """Validate the type of a variable.

        Args:
            variable: The variable to validate.
            expected_type: The expected type of the variable.
            name: The name of the variable.

        Raises:
            ValueError: If the type of the variable does not match the expected type.
        """
        if not isinstance(variable, expected_type):
            raise ValueError(
                f"Invalid type for {name}: Expected {expected_type.__name__}, "
                f"got {type(variable).__name__}."
            )

    def _validate_value_range(
        self,
        value: int | float,
        min_value: int | float,
        max_value: int | float,
        name: str,
    ) -> None:
        """Validate that a value is within a specific range.

        Args:
            value: The value to validate.
            min_value: The minimum value allowed.
            max_value: The maximum value allowed.
            name: The name of the value.

        Raises:
            ValueError: If the value is not within the specified range.
        """
        if not min_value <= value <= max_value:
            raise ValueError(
                f"{name} must be between {min_value} and {max_value}. Got: {value}."
            )

    def _validate_list_of(self, lst: list[Any], expected_type: Any, name: str) -> None:
        """Validate that all items in a list are of a specific type.

        Args:
            lst: The list to validate.
            expected_type: The expected type of the items in the list.
            name: The name of the list.

        Raises:
            ValueError: If any item in the list is not of the expected type.
        """
        if any(not isinstance(item, expected_type) for item in lst):
            raise ValueError(
                f"All items in {name} must be of type {expected_type.__name__}."
            )

    def _validate_inclusion(
        self, value: Any, valid_options: list[Any], name: str
    ) -> None:
        """Validate that a value is included in a list of valid options.

        Args:
            value: The value to validate.
            valid_options: A list of valid options.
            name: The name of the value.

        Raises:
            ValueError: If the value is not in the list of valid options.
        """
        if value not in valid_options:
            valid_str = ", ".join(valid_options)
            raise ValueError(f"Invalid {name}: {value}. Valid options: {valid_str}.")

    def _validate_matching_lengths(
        self, list1: list[Any], list2: list[Any], name1: str, name2: str
    ) -> None:
        """Validate that two lists have the same length.

        Args:
            list1: The first list to compare.
            list2: The second list to compare.
            name1: The name of the first list.
            name2: The name of the second list.

        Raises:
            ValueError: If the lengths of the two lists do not match.
        """
        if len(list1) != len(list2):
            raise ValueError(
                f"{name1} and {name2} must have the same size. "
                f"Lengths: {len(list1)} and {len(list2)}."
            )

    def _validate_dictionary(
        self, dct: dict[str, Any], dct_validation: dict[str, Any], dict_name: str
    ) -> None:
        """Validate a dictionary based on required keys and value types.

        Args:
            dct: The dictionary to validate.
            dct_validation: The dictionary to validate against.
            dict_name: The name of the dictionary.

        Raises:
            ValueError: If the dictionary is missing required keys.

        See Also:
            _validate_inclusion() and _validate_type() for validation details.
        """
        required_keys = list(dct_validation.keys())
        missing_keys = [key for key in required_keys if key not in dct]
        if missing_keys:
            raise ValueError(f"{dict_name} is missing keys: {', '.join(missing_keys)}")
        for key, value in dct.items():
            self._validate_inclusion(key, required_keys, f"{dict_name} key")
            self._validate_type(
                value,
                dct_validation[key][0],
                f"{dict_name}[{key}]",
            )

    def _validate_can_be_updated(
        self, param_name: str, invalid_params: list[str], txt: str
    ) -> None:
        """Validate that a parameter can be updated.

        Args:
            param_name: The name of the parameter to update.
            invalid_params: A list of parameters that cannot be updated.
            txt: A string to include in the error message.

        Raises:
            ValueError: If the parameter cannot be updated.
        """
        if param_name in invalid_params:
            raise ValueError(
                f"Parameter '{param_name}' cannot be updated directly. {txt}"
            )
