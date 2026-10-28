# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting data preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import cast

import pandas as pd

from frequenz.data.microgrid import MicrogridConfig, MicrogridData
from frequenz.lib.notebooks.reporting.data_processing import (
    create_energy_report_df,
    fetch_energy_report_data,
)
from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper


@dataclass
class _DummyMeta:
    """Minimal config metadata stub with only the microgrid id."""

    microgrid_id: int


class _DummyMicrogridConfig:
    """Minimal config stub exposing component type helpers and metadata."""

    def __init__(
        self,
        mapping: dict[str, list[str]],
        formulas: dict[tuple[str, str], str] | None = None,
        microgrid_id: int = 241,
    ) -> None:
        self.mapping = mapping
        self.formulas = formulas or {}
        self.meta = _DummyMeta(microgrid_id)

    def component_types(self) -> list[str]:
        return list(self.mapping.keys())

    def component_type_ids(
        self, component_type: str, component_category: str | None = None
    ) -> list[str]:
        del component_category
        return self.mapping.get(component_type, [])

    def formula(self, component_type: str, metric: str) -> str:
        formula = self.formulas.get((component_type, metric))
        if formula is None:
            raise ValueError(f"{component_type} is missing formula for {metric}")
        return formula


def test_create_energy_report_df_appends_meter_display_names() -> None:
    """Explicit Assets API display names are appended to component labels."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "pv": [-12.0],
            "1179": [-12.0],
        }
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["pv"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"pv": ["1179"]})),
        mapper=ColumnMapper.from_default(locale="en"),
        component_display_names={"1179": "PV Roof Meter"},
        grid_data_source="power",
    )

    assert "PV #1179 - PV Roof Meter" in result.columns


def test_create_energy_report_df_uses_explicit_component_display_names() -> None:
    """Explicit display-name mappings override the Assets API lookup."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "pv": [-12.0],
            "1179": [-12.0],
        }
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["pv"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"pv": ["1179"]})),
        mapper=ColumnMapper.from_default(locale="en"),
        component_display_names={"1179": "Explicit PV Meter"},
        grid_data_source="power",
    )

    assert "PV #1179 - Explicit PV Meter" in result.columns


def test_create_energy_report_df_adds_battery_soc_when_battery_present() -> None:
    """Fetched SOC is included whenever battery is among the component types."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-09 06:30:00", "2026-01-09 06:45:00"],
                utc=True,
            ),
            "grid": [30.0, 25.0],
            "consumption": [35.0, 21.0],
            "battery": [5.0, -4.0],
        }
    )
    soc_df = pd.DataFrame(
        {"battery": [42.0, 53.0]},
        index=pd.to_datetime(
            ["2026-01-09 06:30:00", "2026-01-09 06:45:00"],
            utc=True,
        ),
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["battery"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"battery": []})),
        mapper=ColumnMapper.from_default(locale="en"),
        battery_soc_df=soc_df,
        grid_data_source="power",
    )

    assert "battery_soc_pct" in result.columns
    assert list(result["battery_soc_pct"]) == [42.0, 53.0]


def test_create_energy_report_df_adds_battery_soc_bounds_when_available() -> None:
    """Fetched SOC bounds are included whenever battery is among the component types."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-09 06:30:00", "2026-01-09 06:45:00"],
                utc=True,
            ),
            "grid": [30.0, 25.0],
            "consumption": [35.0, 21.0],
            "battery": [5.0, -4.0],
        }
    )
    soc_df = pd.DataFrame(
        {
            "battery": [42.0, 53.0],
            "battery_soc_lower_bound_pct": [20.0, 25.0],
            "battery_soc_upper_bound_pct": [80.0, 75.0],
        },
        index=pd.to_datetime(
            ["2026-01-09 06:30:00", "2026-01-09 06:45:00"],
            utc=True,
        ),
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["battery"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"battery": []})),
        mapper=ColumnMapper.from_default(locale="en"),
        battery_soc_df=soc_df,
        grid_data_source="power",
    )

    assert list(result["battery_soc_lower_bound_pct"]) == [20.0, 25.0]
    assert list(result["battery_soc_upper_bound_pct"]) == [80.0, 75.0]


def test_create_energy_report_df_keeps_battery_flow_columns() -> None:
    """Derived battery flow columns should survive final report column selection."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-09 06:30:00",
                    "2026-01-09 06:45:00",
                    "2026-01-09 07:00:00",
                    "2026-01-09 07:15:00",
                ],
                utc=True,
            ),
            "grid": [-3.0, 7.0, 1.0, -7.0],
            "pv": [-10.0, -2.0, -1.0, -8.0],
            "consumption": [4.0, 5.0, 4.0, 3.0],
            "battery": [3.0, 4.0, -2.0, -2.0],
        }
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["battery", "pv"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"battery": [], "pv": []})),
        mapper=ColumnMapper.from_default(locale="en"),
        grid_data_source="power",
    )

    assert result["grid_consumption"].tolist() == [0.0, 7.0, 1.0, 0.0]
    assert result["grid_feed_in"].tolist() == [3.0, 0.0, 0.0, 7.0]
    assert result["production_to_battery"].tolist() == [3.0, 0.0, 0.0, 0.0]
    assert result["grid_to_battery"].tolist() == [0.0, 4.0, 0.0, 0.0]
    assert result["battery_to_grid"].tolist() == [0.0, 0.0, 0.0, 2.0]
    assert result["battery_to_consumption"].tolist() == [0.0, 0.0, 2.0, 0.0]


def test_create_energy_report_df_uses_directional_energy_by_default() -> None:
    """Default grid data converts directional energy deltas to report power."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-09 06:30:00", "2026-01-09 06:45:00"], utc=True
            ),
            "grid_energy_consumed": [2.0, 0.0],
            "grid_energy_delivered": [0.0, 1.0],
            "pv": [-4.0, -4.0],
            "consumption": [10.0, 2.0],
        }
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["pv"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"pv": []})),
        mapper=ColumnMapper.from_default(locale="en"),
        resolution=timedelta(minutes=15),
    )

    assert result["grid_consumption"].tolist() == [8.0, 0.0]
    assert result["grid_feed_in"].tolist() == [0.0, 4.0]


async def test_fetch_energy_report_data_uses_energy_only_for_grid_by_default() -> None:
    """Only grid directions use active energy; component data stays power."""

    class _MicrogridDataStub:
        def __init__(self) -> None:
            self.power_calls: list[dict[str, object]] = []
            self.energy_calls: list[dict[str, object]] = []

        async def ac_active_power(self, **kwargs: object) -> pd.DataFrame:
            self.power_calls.append(kwargs)
            return pd.DataFrame({"grid": [3.0], "pv": [-2.0]})

        async def ac_active_energy(self, **kwargs: object) -> pd.DataFrame:
            self.energy_calls.append(kwargs)
            direction = kwargs["direction"]
            values = [1.0] if direction == "consumed" else [0.5]
            return pd.DataFrame({"grid": values})

    mdata = _MicrogridDataStub()
    result = await fetch_energy_report_data(
        cast(MicrogridData, mdata),
        microgrid_id=241,
        component_types=("grid", "pv"),
        start=pd.Timestamp("2026-01-09T06:30:00Z").to_pydatetime(),
        end=pd.Timestamp("2026-01-09T06:45:00Z").to_pydatetime(),
    )

    assert mdata.power_calls[0]["component_types"] == ("grid", "pv")
    assert [call["component_types"] for call in mdata.energy_calls] == [
        ("grid",),
        ("grid",),
    ]
    assert [call["direction"] for call in mdata.energy_calls] == [
        "consumed",
        "delivered",
    ]
    assert result is not None
    assert result.columns.tolist() == [
        "pv",
        "grid_energy_consumed",
        "grid_energy_delivered",
    ]

    power_result = await fetch_energy_report_data(
        cast(MicrogridData, mdata),
        microgrid_id=241,
        component_types=("grid", "pv"),
        start=pd.Timestamp("2026-01-09T06:30:00Z").to_pydatetime(),
        end=pd.Timestamp("2026-01-09T06:45:00Z").to_pydatetime(),
        grid_data_source="power",
    )

    assert len(mdata.power_calls) == 2
    assert len(mdata.energy_calls) == 2
    assert power_result is not None
    assert power_result.columns.tolist() == ["grid", "pv"]


def test_create_energy_report_df_excludes_battery_soc_when_battery_not_requested() -> (
    None
):
    """SOC data is ignored when battery is not among the requested component types."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "pv": [-12.0],
        }
    )
    soc_df = pd.DataFrame(
        {"battery": [42.0]},
        index=pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["pv"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"pv": []})),
        mapper=ColumnMapper.from_default(locale="en"),
        battery_soc_df=soc_df,
        grid_data_source="power",
    )

    assert "battery_soc_pct" not in result.columns


def test_create_energy_report_df_drops_premerged_soc_column() -> None:
    """A raw soc column is ignored without explicit SOC input data."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "consumption": [35.0],
            "battery": [5.0],
            "soc": [42.0],
        }
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["battery"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"battery": []})),
        mapper=ColumnMapper.from_default(locale="en"),
        grid_data_source="power",
    )

    assert "battery_soc_pct" not in result.columns


def test_create_energy_report_df_drops_premerged_soc_when_battery_not_requested() -> (
    None
):
    """A raw soc column is ignored when battery is not requested."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "pv": [-12.0],
            "soc": [42.0],
        }
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["pv"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"pv": []})),
        mapper=ColumnMapper.from_default(locale="en"),
        grid_data_source="power",
    )

    assert "battery_soc_pct" not in result.columns


def test_battery_soc_display_name_is_german_percent_label() -> None:
    """Battery SOC should have the requested German display label."""
    german_mapper = ColumnMapper.from_default(locale="de")
    english_mapper = ColumnMapper.from_default(locale="en")

    assert german_mapper.canonical_to_display["battery_soc_pct"] == "Batterie SOC %"
    assert english_mapper.canonical_to_display["battery_soc_pct"] == "Battery SOC (%)"
