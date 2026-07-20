# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting data preparation helpers."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
from frequenz.gridpool import MicrogridConfig
from pandas.testing import assert_frame_equal

from frequenz.lib.notebooks.reporting.data_processing import (
    build_energy_report,
    create_battery_usecase_df,
    create_energy_report_df,
)
from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper


class _DummyMeta:
    """Minimal config metadata stub with only the microgrid id."""

    def __init__(self, microgrid_id: int) -> None:
        self.microgrid_id = microgrid_id


class _DummyMicrogridConfig:
    """Minimal config stub exposing component type helpers and metadata."""

    def __init__(
        self,
        mapping: dict[str, list[str]],
        microgrid_id: int = 241,
        ctype: dict[str, object] | None = None,
    ) -> None:
        self.mapping = mapping
        self.meta = _DummyMeta(microgrid_id)
        self.ctype = ctype

    def component_types(self) -> list[str]:
        if self.ctype:
            return list(self.ctype.keys())
        return list(self.mapping.keys())

    def component_type_ids(self, component_type: str) -> list[str]:
        return self.mapping.get(component_type, [])


class _DummyComponentConfig:
    """Minimal component config stub with meter/inverter/component groups."""

    def __init__(
        self,
        *,
        meter: list[str] | None = None,
        inverter: list[str] | None = None,
        component: list[str] | None = None,
    ) -> None:
        self.meter = meter
        self.inverter = inverter
        self.component = component


def _as_microgrid_config(mcfg: _DummyMicrogridConfig) -> MicrogridConfig:
    """Cast a lightweight test double to the production config type."""
    return cast(MicrogridConfig, mcfg)


def test_create_battery_usecase_df_builds_expected_columns() -> None:
    """Battery usecase helper derives the expected canonical columns."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
        }
    )
    result = create_battery_usecase_df(energy_report_df)

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_charge": [5.0, 0.0],
            "battery_discharge": [0.0, -4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_create_battery_usecase_df_accepts_custom_input_column_names() -> None:
    """Custom source column names are normalized to the canonical output schema."""
    energy_report_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "load": [35.0, 21.0],
            "grid_load": [30.0, 25.0],
            "battery_flow": [5.0, -4.0],
            "pv_power": [12.0, 10.0],
        }
    )
    result = create_battery_usecase_df(
        energy_report_df,
        timestamp_col="time",
        consumption_col="load",
        grid_consumption_col="grid_load",
        battery_col="battery_flow",
        pv_col="pv_power",
    )

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv": [12.0, 10.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_charge": [5.0, 0.0],
            "battery_discharge": [0.0, -4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_create_battery_usecase_df_requires_configured_input_columns() -> None:
    """Missing required columns should fail clearly."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "consumption": [35.0],
            "grid_consumption": [30.0],
        }
    )

    with pytest.raises(KeyError, match="battery_power_flow"):
        create_battery_usecase_df(energy_report_df)


def test_build_energy_report_includes_meter_names_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meter names are stored as display metadata when fetched from assets."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "pv": [-12.0],
            "1179": [-12.0],
        }
    )
    mapper = ColumnMapper.from_default(locale="en")
    mcfg = _DummyMicrogridConfig({"pv": ["1179"]}, microgrid_id=241)

    monkeypatch.setattr(
        "frequenz.lib.notebooks.reporting.data_processing.get_meter_display_names",
        lambda microgrid_id: {"1179": "PV Roof Meter"},
    )

    result = build_energy_report(
        raw_df,
        component_types=["pv"],
        mcfg=_as_microgrid_config(mcfg),
        mapper=mapper,
    )

    assert "1179" in result.df.columns
    assert result.metadata.display_names == {"1179": "PV Roof Meter"}
    assert result.metadata.ids_by_type == {"pv": ["1179"]}


def test_build_energy_report_accepts_explicit_component_display_names() -> None:
    """Explicit display-name mappings should be stored as display metadata."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "battery": [5.0],
            "1532": [5.0],
        }
    )
    mapper = ColumnMapper.from_default(locale="en")
    mcfg = _DummyMicrogridConfig({"battery": ["1532"]}, microgrid_id=241)

    result = build_energy_report(
        raw_df,
        component_types=["battery"],
        mcfg=_as_microgrid_config(mcfg),
        mapper=mapper,
        component_display_names={"1532": "Battery East"},
    )

    assert "1532" in result.df.columns
    assert result.metadata.display_names == {"1532": "Battery East"}
    assert result.metadata.ids_by_type == {"battery": ["1532"]}


def test_build_energy_report_keeps_inverter_component_columns() -> None:
    """Per-inverter PV columns should survive into the energy report dataframe."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "pv": [-12.0],
            "1134": [-7.0],
            "1135": [-5.0],
        }
    )
    mapper = ColumnMapper.from_default(locale="en")
    mcfg = _DummyMicrogridConfig(
        {"pv": []},
        microgrid_id=241,
        ctype={
            "pv": _DummyComponentConfig(inverter=["1134", "1135"]),
        },
    )

    result = build_energy_report(
        raw_df,
        component_types=["pv"],
        mcfg=_as_microgrid_config(mcfg),
        mapper=mapper,
        component_display_names={},
    )

    assert "1134" in result.df.columns
    assert "1135" in result.df.columns
    assert result.metadata.ids_by_type == {"pv": ["1134", "1135"]}
    assert "pv_asset_production" in result.df.columns


def test_create_energy_report_df_returns_dataframe_by_default() -> None:
    """The public API should continue to return only the dataframe by default."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "battery": [5.0],
            "1532": [5.0],
        }
    )
    mapper = ColumnMapper.from_default(locale="en")
    mcfg = _DummyMicrogridConfig({"battery": ["1532"]}, microgrid_id=241)

    result = create_energy_report_df(
        raw_df,
        component_types=["battery"],
        mcfg=_as_microgrid_config(mcfg),
        mapper=mapper,
    )

    assert isinstance(result, pd.DataFrame)
    assert "1532" in result.columns


def test_create_energy_report_df_skips_meter_lookup_without_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default dataframe-only calls should not fetch display names."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "battery": [5.0],
            "1532": [5.0],
        }
    )
    mapper = ColumnMapper.from_default(locale="en")
    mcfg = _DummyMicrogridConfig({"battery": ["1532"]}, microgrid_id=241)

    def fail_lookup(_: int) -> dict[str, str]:
        raise AssertionError("meter lookup should not run")

    monkeypatch.setattr(
        "frequenz.lib.notebooks.reporting.data_processing.get_meter_display_names",
        fail_lookup,
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["battery"],
        mcfg=_as_microgrid_config(mcfg),
        mapper=mapper,
    )

    assert isinstance(result, pd.DataFrame)
    assert "1532" in result.columns


def test_create_energy_report_df_skips_component_types_without_ids() -> None:
    """Misconfigured component types should not fail dataframe creation."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "pv": [-12.0],
            "battery": [5.0],
            "1532": [5.0],
        }
    )
    mapper = ColumnMapper.from_default(locale="en")
    mcfg = _DummyMicrogridConfig(
        {
            "grid": [],
            "pv": [],
            "battery": ["1532"],
        },
        microgrid_id=241,
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["grid", "pv", "battery"],
        mcfg=_as_microgrid_config(mcfg),
        mapper=mapper,
    )

    assert isinstance(result, pd.DataFrame)
    assert "1532" in result.columns
    assert "timestamp" in result.columns


def test_build_energy_report_can_skip_meter_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata lookups can be disabled while keeping the stable return shape."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "battery": [5.0],
            "1532": [5.0],
        }
    )
    mapper = ColumnMapper.from_default(locale="en")
    mcfg = _DummyMicrogridConfig({"battery": ["1532"]}, microgrid_id=241)

    def fail_lookup(_: int) -> dict[str, str]:
        raise AssertionError("meter lookup should not run")

    monkeypatch.setattr(
        "frequenz.lib.notebooks.reporting.data_processing.get_meter_display_names",
        fail_lookup,
    )

    result = build_energy_report(
        raw_df,
        component_types=["battery"],
        mcfg=_as_microgrid_config(mcfg),
        mapper=mapper,
        include_component_metadata=False,
    )

    assert isinstance(result.df, pd.DataFrame)
    assert result.metadata.ids_by_type == {"battery": ["1532"]}
