# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting helpers."""

from __future__ import annotations

from typing import cast

from frequenz.gridpool import MicrogridConfig

from frequenz.lib.notebooks.reporting.utils.helpers import get_component_ids


class _DummyMicrogridConfig:
    """Minimal config stub exposing component type helpers."""

    def __init__(self, mapping: dict[str, list[str] | Exception]) -> None:
        self.mapping = mapping

    def component_types(self) -> list[str]:
        return list(self.mapping.keys())

    def component_type_ids(self, component_type: str) -> list[str]:
        result = self.mapping[component_type]
        if isinstance(result, Exception):
            raise result
        return result


def _as_microgrid_config(mcfg: _DummyMicrogridConfig) -> MicrogridConfig:
    """Cast a lightweight test double to the production config type."""
    return cast(MicrogridConfig, mcfg)


def test_get_component_ids_skips_types_without_configured_ids() -> None:
    """Missing ids in gridpool config should be treated as no components."""
    mcfg = _DummyMicrogridConfig({"grid": ValueError("No IDs available")})

    result = get_component_ids(_as_microgrid_config(mcfg), "grid")

    assert result == []


def test_get_component_ids_reraises_unrelated_value_errors() -> None:
    """Only the known missing-id sentinel should be swallowed."""
    mcfg = _DummyMicrogridConfig({"grid": ValueError("unexpected config issue")})

    try:
        get_component_ids(_as_microgrid_config(mcfg), "grid")
    except ValueError as err:
        assert str(err) == "unexpected config issue"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError to be reraised")
