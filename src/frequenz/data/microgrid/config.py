# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Configuration for microgrids."""

import re
import tomllib
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

ComponentType = Literal["grid", "pv", "battery", "consumption", "chp"]
"""Valid component types."""

ComponentCategory = Literal["meter", "inverter", "component"]
"""Valid component categories."""


@dataclass
class ComponentTypeConfig:
    """Configuration of a microgrid component type."""

    component_type: ComponentType
    """Type of the component."""

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
        if self.component_type == "battery" and "BATTERY_SOC_PCT" not in self.formula:
            if self.component:
                cids = self.component
                form = "+".join([f"#{cid}" for cid in cids])
                form = f"({form})/({len(cids)})"
                self.formula["BATTERY_SOC_PCT"] = form

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
                raise ValueError(
                    f"{metric} does not have a formula for {self.component_type}"
                )
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

        raise ValueError(f"No IDs available for {self.component_type}")

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


@dataclass(frozen=True)
class WindConfig:
    """Configuration of a wind turbine in a microgrid."""

    turbine_model: str | None = None
    """Model name of the wind turbine."""

    rated_power: float | None = None
    """Rated power of the wind turbine in Watt."""

    turbine_height: float | None = None
    """Height of the wind turbine in meters."""


@dataclass(frozen=True)
class BatteryConfig:
    """Configuration of a battery in a microgrid."""

    capacity: float | None = None
    """Capacity of the battery in Wh."""


@dataclass(frozen=True)
class AssetsConfig:
    """Configuration of the assets in a microgrid."""

    pv: dict[str, PVConfig] | None = None
    """Configuration of the PV system."""

    wind: dict[str, WindConfig] | None = None
    """Configuration of the wind turbines."""

    battery: dict[str, BatteryConfig] | None = None
    """Configuration of the batteries."""


@dataclass(frozen=True)
class Metadata:
    """Metadata for a microgrid."""

    name: str | None = None
    """Name of the microgrid."""

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

    _metadata: Metadata
    """Metadata of the microgrid."""

    _assets_cfg: AssetsConfig
    """Configuration of the assets in the microgrid."""

    _component_types_cfg: dict[str, ComponentTypeConfig]
    """Mapping of component category types to ac power component config."""

    def __init__(self, config_dict: dict[str, Any]) -> None:
        """Initialize the microgrid configuration.

        Args:
            config_dict: Dictionary with component type as key and config as value.
        """
        self._metadata = Metadata(**(config_dict.get("meta") or {}))

        self._assets_cfg = AssetsConfig(
            pv=config_dict.get("pv") or {},
            wind=config_dict.get("wind") or {},
            battery=config_dict.get("battery") or {},
        )

        self._component_types_cfg = {
            ctype: ComponentTypeConfig(component_type=cast(ComponentType, ctype), **cfg)
            for ctype, cfg in config_dict.get("ctype", {}).items()
            if ComponentTypeConfig.is_valid_type(ctype)
        }

    @property
    def meta(self) -> Metadata:
        """Return the metadata of the microgrid."""
        return self._metadata

    @property
    def assets(self) -> AssetsConfig:
        """Return the assets configuration of the microgrid."""
        return self._assets_cfg

    def component_types(self) -> list[str]:
        """Get a list of all component types in the configuration."""
        return list(self._component_types_cfg.keys())

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
        cfg = self._component_types_cfg.get(component_type)
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
        cfg = self._component_types_cfg.get(component_type)
        if not cfg:
            raise ValueError(f"{component_type} not found in config.")
        if cfg.formula is None:
            raise ValueError(f"No formula set for {component_type}")
        formula = cfg.formula.get(metric)
        if not formula:
            raise ValueError(f"{component_type} is missing formula for {metric}")

        return formula

    @staticmethod
    def load_configs(*paths: str) -> dict[str, "MicrogridConfig"]:
        """Load multiple microgrid configurations from a file.

        Configs for a single microgrid are expected to be in a single file.
        Later files with the same microgrid ID will overwrite the previous configs.

        Args:
            *paths: Path(es) to the config file(s).

        Returns:
            Dictionary of single microgrid formula configs with microgrid IDs as keys.
        """
        microgrid_configs = {}
        for config_path in paths:
            with open(config_path, "rb") as f:
                cfg_dict = tomllib.load(f)
                for microgrid_id, mcfg in cfg_dict.items():
                    microgrid_configs[microgrid_id] = MicrogridConfig(mcfg)
        return microgrid_configs
