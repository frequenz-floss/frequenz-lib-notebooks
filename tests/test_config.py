# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for the frequenz.lib.notebooks.config module."""

from typing import Any, cast

import pytest
from pytest_mock import MockerFixture

from frequenz.lib.notebooks.config import ComponentTypeConfig, MicrogridConfig

VALID_CONFIG: dict[str, dict[str, Any]] = {
    "1": {
        "meta": {"name": "Test Grid", "gid": 1},
        "ctype": {
            "pv": {"meter": [101, 102], "formula": "AC_ACTIVE_POWER"},
            "battery": {
                "inverter": [201, 202, 203],
                "component": [301, 302, 303, 304, 305, 306],
            },
        },
        "pv": {
            "PV1": {"peak_power": 5000, "rated_power": 4500},
            "PV2": {"peak_power": 8000, "rated_power": 7000},
        },
        "battery": {
            "BAT1": {"capacity": 10000},
            "BAT2": {"capacity": 12000},
            "BAT3": {"capacity": 8000},
            "BAT4": {"capacity": 15000},
            "BAT5": {"capacity": 20000},
            "BAT6": {},
        },
    },
}


@pytest.fixture
def valid_microgrid_config() -> MicrogridConfig:
    """Fixture to provide a valid MicrogridConfig instance."""
    return MicrogridConfig(VALID_CONFIG["1"])


def test_is_valid_type() -> None:
    """Test the validation of component types."""
    assert ComponentTypeConfig.is_valid_type("pv")
    assert not ComponentTypeConfig.is_valid_type("unknown")


def test_component_type_config_cids() -> None:
    """Test the retrieval of component IDs for various configurations."""
    config = ComponentTypeConfig(component_type="pv", meter=[1, 2, 3])
    assert config.cids() == [1, 2, 3]

    config = ComponentTypeConfig(component_type="battery", inverter=[4, 5])
    assert config.cids() == [4, 5]

    with pytest.raises(ValueError):
        config = ComponentTypeConfig(component_type="grid")
        config.cids()


def test_component_type_config_default_formula() -> None:
    """Test that the default formula is generated correctly."""
    config = ComponentTypeConfig(component_type="pv", meter=[1, 2])
    assert config._default_formula() == "#1+#2"  # pylint: disable=protected-access


def test_component_type_config_has_formula_for() -> None:
    """Test whether a component type has a valid formula for a metric."""
    config = ComponentTypeConfig(component_type="pv", formula="AC_ACTIVE_POWER")
    assert config.has_formula_for("AC_ACTIVE_POWER")
    assert not config.has_formula_for("INVALID_METRIC")


def test_microgrid_config_init(valid_microgrid_config: MicrogridConfig) -> None:
    """Test initialisation of MicrogridConfig with valid configuration data."""
    assert valid_microgrid_config.meta.name == "Test Grid"
    pv_config = valid_microgrid_config.assets.pv
    if pv_config:
        _assert_optional_field(
            cast(dict[str, float], pv_config["PV1"]).get("peak_power"), 5000
        )


def test_microgrid_config_component_types(
    valid_microgrid_config: MicrogridConfig,
) -> None:
    """Test retrieval of all component types in the configuration."""
    assert valid_microgrid_config.component_types() == ["pv", "battery"]


def test_microgrid_config_component_type_ids(
    valid_microgrid_config: MicrogridConfig,
) -> None:
    """Test retrieval of component IDs for a given component type."""
    assert valid_microgrid_config.component_type_ids("pv") == [101, 102]
    assert valid_microgrid_config.component_type_ids("battery") == [201, 202, 203]
    assert valid_microgrid_config.component_type_ids("battery", "inverter") == [
        201,
        202,
        203,
    ]
    assert valid_microgrid_config.component_type_ids("battery", "component") == [
        301,
        302,
        303,
        304,
        305,
        306,
    ]

    with pytest.raises(ValueError):
        valid_microgrid_config.component_type_ids("unknown")


def test_microgrid_config_formula(valid_microgrid_config: MicrogridConfig) -> None:
    """Test retrieval of formula for a given component type and metric."""
    assert valid_microgrid_config.formula("pv", "AC_ACTIVE_POWER") == "AC_ACTIVE_POWER"

    with pytest.raises(ValueError):
        valid_microgrid_config.formula("pv", "INVALID_METRIC")


def test_load_configs(mocker: MockerFixture) -> None:
    """Test loading configurations for multiple microgrids from mock TOML files."""
    toml_data = """
    1.meta.name = "Test Grid"
    1.meta.gid = 1
    1.ctype.pv.meter = [101, 102]
    1.ctype.pv.formula = "AC_ACTIVE_POWER"
    1.ctype.battery.inverter = [201, 202, 203]
    1.ctype.battery.component = [301, 302, 303, 304, 305, 306]
    1.pv.PV1.peak_power = 5000
    1.pv.PV1.rated_power = 4500
    1.pv.PV2peak_power = 8000
    1.pv.PV2rated_power = 7000
    1.battery.BAT1.capacity = 10000
    """
    mocker.patch("builtins.open", mocker.mock_open(read_data=toml_data.encode("utf-8")))
    configs = MicrogridConfig.load_configs("mock_path.toml")

    assert "1" in configs
    assert configs["1"].meta.name == "Test Grid"
    pv_config = configs["1"].assets.pv
    battery_config = configs["1"].assets.battery
    if pv_config and battery_config:
        _assert_optional_field(
            cast(dict[str, float], pv_config["PV1"]).get("peak_power"), 5000
        )
        _assert_optional_field(
            cast(dict[str, float], battery_config["BAT1"]).get("capacity"), 10000
        )


def _assert_optional_field(value: float | None, expected: float) -> None:
    """Validate an optional field.

    Args:
        value: The optional field value to check.
        expected: The expected value to assert if `value` is not None.
    """
    if value is not None:
        assert value == expected
