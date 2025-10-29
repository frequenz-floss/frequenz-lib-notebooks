# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Configuration for microgrids."""

import logging
import re
import tomllib
from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, Type, cast, get_args

from marshmallow import Schema
from marshmallow_dataclass import dataclass

_logger = logging.getLogger(__name__)

ComponentType = Literal["grid", "pv", "battery", "consumption", "chp", "ev"]
"""Valid component types."""

ComponentCategory = Literal["meter", "inverter", "component"]
"""Valid component categories."""


@dataclass
class ComponentTypeConfig:
    """Configuration of a microgrid component type."""

    meter: list[int] | None = None
    """List of meter IDs for this component."""

    inverter: list[int] | None = None
    """List of inverter IDs for this component."""

    component: list[int] | None = None
    """List of component IDs for this component."""

    formula: dict[str, str] | None = None
    """Formula to calculate the power of this component."""

    def __post_init__(self) -> None:
        """Set the default formula if none is provided."""
        self.formula = self.formula or {}
        if "AC_ACTIVE_POWER" not in self.formula:
            self.formula["AC_ACTIVE_POWER"] = "+".join(
                [f"#{cid}" for cid in self._default_cids()]
            )

    def cids(self, metric: str = "") -> list[int]:
        """Get component IDs for this component.

        By default, the meter IDs are returned if available, otherwise the inverter IDs.
        For components without meters or inverters, the component IDs are returned.

        If a metric is provided, the component IDs are extracted from the formula.

        Args:
            metric: Metric name of the formula.

        Returns:
            List of component IDs for this component.

        Raises:
            ValueError: If the metric is not supported or improperly set.
        """
        if metric:
            if not isinstance(self.formula, dict):
                raise ValueError("Formula must be a dictionary.")
            formula = self.formula.get(metric)
            if not formula:
                raise ValueError(f"{metric} does not have a formula")
            # Extract component IDs from the formula which are given as e.g. #123
            pattern = r"#(\d+)"
            return [int(e) for e in re.findall(pattern, self.formula[metric])]

        return self._default_cids()

    def _default_cids(self) -> list[int]:
        """Get the default component IDs for this component.

        If available, the meter IDs are returned, otherwise the inverter IDs.
        For components without meters or inverters, the component IDs are returned.

        Returns:
            List of component IDs for this component.

        Raises:
            ValueError: If no IDs are available.
        """
        if self.meter:
            return self.meter
        if self.inverter:
            return self.inverter
        if self.component:
            return self.component

        raise ValueError("No IDs available")

    @classmethod
    def is_valid_type(cls, ctype: str) -> bool:
        """Check if `ctype` is a valid enum value."""
        return ctype in get_args(ComponentType)


@dataclass(frozen=True)
class PVConfig:
    """Configuration of a PV system in a microgrid."""

    peak_power: float | None = None
    """Peak power of the PV system in Watt."""

    rated_power: float | None = None
    """Rated power of the inverters in Watt."""

    curtailable: bool = False
    """Flag to indicate if PV system can be curtailed."""


@dataclass(frozen=True)
class WindConfig:
    """Configuration of a wind turbine in a microgrid."""

    turbine_model: str | None = None
    """Model name of the wind turbine."""

    rated_power: float | None = None
    """Rated power of the wind turbine in Watt."""

    turbine_height: float | None = None
    """Height of the wind turbine in meters."""

    number_of_turbines: int = 1
    """Number of wind turbines."""

    hellmann_exponent: float | None = None
    """Hellmann exponent for wind speed extrapolation. See: https://w.wiki/FMw9"""

    longitude: float | None = None
    """Geographic longitude of the wind turbine."""

    latitude: float | None = None
    """Geographic latitude of the wind turbine."""


@dataclass(frozen=True)
class BatteryConfig:
    """Configuration of a battery in a microgrid."""

    capacity: float | None = None
    """Capacity of the battery in Wh."""


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class Metadata:
    """Metadata for a microgrid."""

    microgrid_id: int
    """ID of the microgrid."""

    name: str | None = None
    """Name of the microgrid."""

    enterprise_id: int | None = None
    """Enterprise ID of the microgrid."""

    gid: int | None = None
    """Gridpool ID of the microgrid."""

    delivery_area: str | None = None
    """Delivery area of the microgrid."""

    latitude: float | None = None
    """Geographic latitude of the microgrid."""

    longitude: float | None = None
    """Geographic longitude of the microgrid."""

    altitude: float | None = None
    """Geographic altitude of the microgrid."""


@dataclass
class MicrogridConfig:
    """Configuration of a microgrid."""

    meta: Metadata
    """Metadata of the microgrid."""

    pv: dict[str, PVConfig] | None = None
    """Configuration of the PV system."""

    wind: dict[str, WindConfig] | None = None
    """Configuration of the wind turbines."""

    battery: dict[str, BatteryConfig] | None = None
    """Configuration of the batteries."""

    ctype: dict[str, ComponentTypeConfig] = field(default_factory=dict)
    """Mapping of component category types to ac power component config."""

    def component_types(self) -> list[str]:
        """Get a list of all component types in the configuration."""
        return list(self.ctype.keys())

    def component_type_ids(
        self,
        component_type: str,
        component_category: str | None = None,
        metric: str = "",
    ) -> list[int]:
        """Get a list of all component IDs for a component type.

        Args:
            component_type: Component type to be aggregated.
            component_category: Specific category of component IDs to retrieve
                (e.g., "meter", "inverter", or "component"). If not provided,
                the default logic is used.
            metric: Metric name of the formula if CIDs should be extracted from the formula.

        Returns:
            List of component IDs for this component type.

        Raises:
            ValueError: If the component type is unknown.
            KeyError: If `component_category` is invalid.
        """
        cfg = self.ctype.get(component_type)
        if not cfg:
            raise ValueError(f"{component_type} not found in config.")

        if component_category:
            valid_categories = get_args(ComponentCategory)
            if component_category not in valid_categories:
                raise KeyError(
                    f"Invalid component category: {component_category}. "
                    f"Valid categories are {valid_categories}"
                )
            category_ids = cast(list[int], getattr(cfg, component_category, []))
            return category_ids

        return cfg.cids(metric)

    def formula(self, component_type: str, metric: str) -> str:
        """Get the formula for a component type.

        Args:
            component_type: Component type to be aggregated.
            metric: Metric to be aggregated.

        Returns:
            Formula to be used for this aggregated component as string.

        Raises:
            ValueError: If the component type is unknown or formula is missing.
        """
        cfg = self.ctype.get(component_type)
        if not cfg:
            raise ValueError(f"{component_type} not found in config.")
        if cfg.formula is None:
            raise ValueError(f"No formula set for {component_type}")
        formula = cfg.formula.get(metric)
        if not formula:
            raise ValueError(f"{component_type} is missing formula for {metric}")

        return formula

    Schema: ClassVar[Type[Schema]] = Schema

    @classmethod
    def _load_table_entries(cls, data: dict[str, Any]) -> dict[str, Self]:
        """Load microgrid configurations from table entries.

        Args:
            data: The loaded TOML data.

        Returns:
            A dict mapping microgrid IDs to MicrogridConfig instances.

        Raises:
            ValueError: If top-level keys are not numeric microgrid IDs
                or if there is a microgrid ID mismatch.
            TypeError: If microgrid data is not a dict.
        """
        if not all(str(k).isdigit() for k in data.keys()):
            raise ValueError("All top-level keys must be numeric microgrid IDs.")

        mgrids = {}
        for mid, entry in data.items():
            if not mid.isdigit():
                raise ValueError(
                    f"Table reader: Microgrid ID key must be numeric, got {mid}"
                )
            if not isinstance(entry, dict):
                raise TypeError("Table reader: Each microgrid entry must be a dict")

            mgrid = cls.Schema().load(entry)
            if mgrid.meta is None or mgrid.meta.microgrid_id is None:
                raise ValueError(
                    "Table reader: Each microgrid entry must have a meta.microgrid_id"
                )
            if int(mgrid.meta.microgrid_id) != int(mid):
                raise ValueError(
                    f"Table reader: Microgrid ID mismatch: key {mid} != {mgrid.meta.microgrid_id}"
                )

            mgrids[mid] = mgrid

        return mgrids

    @classmethod
    def load_from_file(cls, config_path: Path) -> dict[int, Self]:
        """
        Load and validate configuration settings from a TOML file.

        Args:
            config_path: the path to the TOML configuration file.

        Returns:
            A dict mapping microgrid IDs to MicrogridConfig instances.
        """
        with config_path.open("rb") as f:
            data = tomllib.load(f)

        assert isinstance(data, dict)

        return cls._load_table_entries(data)

    @staticmethod
    def load_configs(
        microgrid_config_files: str | Path | list[str | Path] | None = None,
        microgrid_config_dir: str | Path | None = None,
    ) -> dict[str, "MicrogridConfig"]:
        """Load multiple microgrid configurations from a file.

        Configs for a single microgrid are expected to be in a single file.
        Later files with the same microgrid ID will overwrite the previous configs.

        Args:
            microgrid_config_files: Path to a single microgrid config file or list of paths.
            microgrid_config_dir: Directory containing multiple microgrid config files.

        Returns:
            Dictionary of single microgrid formula configs with microgrid IDs as keys.

        Raises:
            ValueError: If no config files or dir is provided, or if no config files are found.
        """
        if microgrid_config_files is None and microgrid_config_dir is None:
            raise ValueError(
                "No microgrid config path or directory provided. "
                "Please provide at least one."
            )

        config_files: list[Path] = []

        if microgrid_config_files:
            if isinstance(microgrid_config_files, str):
                config_files = [Path(microgrid_config_files)]
            elif isinstance(microgrid_config_files, Path):
                config_files = [microgrid_config_files]
            elif isinstance(microgrid_config_files, list):
                config_files = [Path(f) for f in microgrid_config_files]

        if microgrid_config_dir:
            if Path(microgrid_config_dir).is_dir():
                config_files += list(Path(microgrid_config_dir).glob("*.toml"))
            else:
                raise ValueError(
                    f"Microgrid config directory {microgrid_config_dir} "
                    "is not a directory"
                )

        if len(config_files) == 0:
            raise ValueError(
                "No microgrid config files found. "
                "Please provide at least one valid config file."
            )

        microgrid_configs: dict[str, "MicrogridConfig"] = {}

        for config_path in config_files:
            if not config_path.is_file():
                _logger.warning("Config path %s is not a file, skipping.", config_path)
                continue

            mcfgs = MicrogridConfig.load_from_file(config_path)
            microgrid_configs.update({str(key): value for key, value in mcfgs.items()})

        return microgrid_configs
