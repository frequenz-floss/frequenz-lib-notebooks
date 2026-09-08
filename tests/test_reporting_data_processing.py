# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting data preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from frequenz.data.microgrid import MicrogridConfig
from frequenz.lib.notebooks.reporting.data_processing import create_energy_report_df
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
    )

    assert "PV #1179 - Explicit PV Meter" in result.columns


def test_create_energy_report_df_adds_battery_soc_when_formula_exists() -> None:
    """Fetched SOC is included for battery reports with SOC formulas."""
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
        mcfg=cast(
            MicrogridConfig,
            _DummyMicrogridConfig(
                {"battery": []},
                formulas={("battery", "BATTERY_SOC_PCT"): "(#1337 + #1339)/2"},
            ),
        ),
        mapper=ColumnMapper.from_default(locale="en"),
        battery_soc_df=soc_df,
    )

    assert "battery_soc_pct" in result.columns
    assert list(result["battery_soc_pct"]) == [42.0, 53.0]


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
    )

    assert result["production_to_battery"].tolist() == [3.0, 0.0, 0.0, 0.0]
    assert result["grid_to_battery"].tolist() == [0.0, 4.0, 0.0, 0.0]
    assert result["battery_to_grid"].tolist() == [0.0, 0.0, 0.0, 2.0]
    assert result["battery_to_consumption"].tolist() == [0.0, 0.0, 2.0, 0.0]


def test_create_energy_report_df_excludes_battery_soc_without_formula() -> None:
    """SOC data is ignored when the config has no battery SOC formula."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "consumption": [35.0],
            "battery": [5.0],
        }
    )
    soc_df = pd.DataFrame(
        {"battery": [42.0]},
        index=pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["battery"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"battery": []})),
        mapper=ColumnMapper.from_default(locale="en"),
        battery_soc_df=soc_df,
    )

    assert "battery_soc_pct" not in result.columns


def test_create_energy_report_df_keeps_premerged_soc_column() -> None:
    """A pre-merged raw ``soc`` column is kept as canonical battery SOC."""
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
        mcfg=cast(
            MicrogridConfig,
            _DummyMicrogridConfig(
                {"battery": []},
                formulas={("battery", "BATTERY_SOC_PCT"): "(#1337 + #1339)/2"},
            ),
        ),
        mapper=ColumnMapper.from_default(locale="en"),
    )

    assert "battery_soc_pct" in result.columns
    assert result.iloc[0]["battery_soc_pct"] == 42.0


def test_create_energy_report_df_drops_premerged_soc_without_formula() -> None:
    """A raw ``soc`` column is ignored without a battery SOC formula."""
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
    )

    assert "battery_soc_pct" not in result.columns


def test_battery_soc_display_name_is_german_percent_label() -> None:
    """Battery SOC should have the requested German display label."""
    german_mapper = ColumnMapper.from_default(locale="de")
    english_mapper = ColumnMapper.from_default(locale="en")

    assert german_mapper.canonical_to_display["battery_soc_pct"] == "Batterie SOC %"
    assert english_mapper.canonical_to_display["battery_soc_pct"] == "Battery SOC (%)"
