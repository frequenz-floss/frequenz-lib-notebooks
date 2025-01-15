# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Configuration for microgrids."""

import tomllib
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

ComponentType = Literal["grid", "pv", "battery", "load", "chp"]
"""Valid component types."""


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

    formula: str = ""
    """Formula to calculate the power of this component."""

    def __post_init__(self) -> None:
        """Set the default formula if none is provided."""
        if not self.formula:
            self.formula = self._default_formula()

    def cids(self) -> list[int]:
        """Get component IDs for this component.

        By default, the meter IDs are returned if available, otherwise the inverter IDs.
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

    def _default_formula(self) -> str:
        """Return the default formula for this component."""
        return "+".join([f"#{cid}" for cid in self.cids()])

    def has_formula_for(self, metric: str) -> bool:
        """Return whether this formula is valid for a metric."""
        return metric in ["AC_ACTIVE_POWER", "AC_REACTIVE_POWER"]

    @classmethod
    def is_valid_type(cls, ctype: str) -> bool:
        """Check if `ctype` is a valid enum value."""
        return ctype in get_args(ComponentType)


@dataclass(frozen=True)
class Metadata:
    """Metadata for a microgrid."""

    name: str | None = None
    """Name of the microgrid."""

    gid: int | None = None
    """Gridpool ID of the microgrid."""

    delivery_area: str | None = None
    """Delivery area of the microgrid."""


@dataclass
class MicrogridConfig:
    """Configuration of a microgrid."""

    _metadata: Metadata
    """Metadata of the microgrid."""

    _component_types_cfg: dict[str, ComponentTypeConfig]
    """Mapping of component category types to ac power component config."""

    def __init__(self, config_dict: dict[str, Any]) -> None:
        """Initialize the microgrid configuration.

        Args:
            config_dict: Dictionary with component type as key and config as value.
        """
        self._metadata = Metadata(**(config_dict.get("meta") or {}))

        self._component_types_cfg = {
            ctype: ComponentTypeConfig(component_type=cast(ComponentType, ctype), **cfg)
            for ctype, cfg in config_dict["ctype"].items()
            if ComponentTypeConfig.is_valid_type(ctype)
        }

    @property
    def meta(self) -> Metadata:
        """Return the metadata of the microgrid."""
        return self._metadata

    def component_types(self) -> list[str]:
        """Get a list of all component types in the configuration."""
        return list(self._component_types_cfg.keys())

    def component_type_ids(self, component_type: str) -> list[int]:
        """Get a list of all component IDs for a component type.

        Args:
            component_type: Component type to be aggregated.

        Returns:
            List of component IDs for this component type.

        Raises:
            ValueError: If the component type is unknown.
        """
        cfg = self._component_types_cfg.get(component_type)
        if not cfg:
            raise ValueError(f"{component_type} not found in config.")

        return cfg.cids()

    def formula(self, component_type: str, metric: str) -> str:
        """Get the formula for a component type.

        Args:
            component_type: Component type to be aggregated.
            metric: Metric to be aggregated.

        Returns:
            Formula to be used for this aggregated component as string.

        Raises:
            ValueError: If the component type is unknown.
        """
        cfg = self._component_types_cfg.get(component_type)
        if not cfg:
            raise ValueError(f"{component_type} not found in config.")

        if not cfg.has_formula_for(metric):
            raise ValueError(f"{metric} not supported for {component_type}")

        return cfg.formula

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
