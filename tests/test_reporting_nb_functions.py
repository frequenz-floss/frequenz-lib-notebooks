# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting notebook utility functions."""

from __future__ import annotations

from datetime import timedelta
from typing import cast

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from frequenz.data.microgrid import MicrogridConfig
from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper
from frequenz.lib.notebooks.reporting.utils.reporting_nb_functions import (
    aggregate_metrics,
    assemble_component_analysis,
    build_component_analysis,
    build_overview_df,
    compute_energy_summary,
)


class _DummyComponentConfig:
    """Minimal component config stub with id groups."""

    def __init__(
        self,
        *,
        meter: list[int] | None = None,
        inverter: list[int] | None = None,
    ) -> None:
        self.meter = meter
        self.inverter = inverter


class _DummyMicrogridConfig:
    """Minimal microgrid config stub exposing component type helpers."""

    def __init__(self, ctype: dict[str, _DummyComponentConfig]) -> None:
        self.ctype = ctype

    def component_type_ids(
        self, component_type: str, component_category: str | None = None
    ) -> list[int]:
        config = self.ctype.get(component_type)
        if config is None:
            raise ValueError(f"{component_type} not found in config.")
        if component_category is None:
            raise ValueError("component_category is required in this stub")
        return cast(list[int], getattr(config, component_category, None)) or []


def test_build_component_analysis_selects_all_components_and_melts() -> None:
    """All matching component columns are reshaped into a long-format table."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "Battery #1": [2.0, -1.0],
            "Battery #2": [-3.0, 4.0],
            "PV #1": [5.0, 6.0],
        }
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["All"],
        component_label="Battery",
        value_col_name="battery_power_flow",
    )

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                ]
            ),
            "Battery": ["#1", "#1", "#2", "#2"],
            "battery_power_flow": [2.0, -1.0, -3.0, 4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_build_component_analysis_returns_display_names_when_available() -> None:
    """Display names are surfaced instead of raw `#id` labels when present."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "PV #1179 - PV Roof Meter": [1.2, 1.5],
            "PV #1188 - PV Yard Meter": [0.7, 0.8],
        }
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["All"],
        component_label="PV",
        value_col_name="pv_asset_production",
    )

    assert result["PV"].tolist() == [
        "PV #1179 - PV Roof Meter",
        "PV #1179 - PV Roof Meter",
        "PV #1188 - PV Yard Meter",
        "PV #1188 - PV Yard Meter",
    ]
    assert result["pv_asset_production"].tolist() == [1.2, 1.5, 0.7, 0.8]


def test_build_component_analysis_can_select_by_display_name() -> None:
    """A display-name selection should match the fully labeled component column."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "PV #1179 - PV Roof Meter": [1.2],
            "PV #1188 - PV Yard Meter": [0.7],
        }
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["PV #1179 - PV Roof Meter"],
        component_label="PV",
        value_col_name="pv_asset_production",
    )

    assert result["PV"].tolist() == ["PV #1179 - PV Roof Meter"]
    assert result["pv_asset_production"].tolist() == [1.2]


def test_build_component_analysis_accepts_selection_generator() -> None:
    """Generator selections should not be consumed before component matching."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "PV #1179 - PV Roof Meter": [1.2],
            "PV #1188 - PV Yard Meter": [0.7],
        }
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=(item for item in ["PV #1179 - PV Roof Meter"]),
        component_label="PV",
        value_col_name="pv_asset_production",
    )

    assert result["PV"].tolist() == ["PV #1179 - PV Roof Meter"]
    assert result["pv_asset_production"].tolist() == [1.2]


def test_build_component_analysis_returns_empty_when_columns_missing() -> None:
    """A missing component selection should return a typed empty frame."""
    energy_report_df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-09 06:30:00"]), "Battery #1": [2.0]}
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["#9"],
        component_label="Battery",
        value_col_name="battery_power_flow",
    )

    assert_frame_equal(
        result,
        pd.DataFrame(columns=["timestamp", "Battery", "battery_power_flow"]),
    )


def test_assemble_component_analysis_scales_and_truncates_component_sum() -> None:
    """Scaled values are returned and the optional truncated sum is respected."""
    mapper = ColumnMapper.from_default(locale="en")
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "Battery #1": [2.0, -1.0],
            "Battery #2": [-3.0, 4.0],
        }
    )

    result_df, component_sum, filter_text = assemble_component_analysis(
        component_filter=["Alle"],
        component_key="battery",
        component_types=["battery", "pv"],
        energy_report_df=energy_report_df,
        timestep_hours=0.5,
        mapper=mapper,
        component_label="Battery",
        value_col_name="battery_power_flow",
        invert_sign=True,
        trunc_values=True,
    )

    expected_df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                [
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                ]
            ),
            "Battery": ["#1", "#1", "#2", "#2"],
            "Battery Power Flow": [-1.0, 0.5, 1.5, -2.0],
        }
    )

    assert_frame_equal(result_df, expected_df)
    assert component_sum == 2.0
    assert filter_text == "All"


def test_assemble_component_analysis_coerces_object_values_to_numeric() -> None:
    """Object-typed component values should still be scaled and rounded."""
    analyse_df, component_sum, filter_text = assemble_component_analysis(
        component_filter=["#1179"],
        component_key="pv",
        component_types=["pv"],
        energy_report_df=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
                "PV #1179 - PV Roof Meter": ["-1.2345"],
            }
        ),
        timestep_hours=1.0,
        mapper=ColumnMapper.from_default(locale="en"),
        component_label="PV",
        value_col_name="pv_asset_production",
        invert_sign=True,
        trunc_values=True,
    )

    assert analyse_df["PV"].tolist() == ["PV #1179 - PV Roof Meter"]
    assert analyse_df["PV-Production"].tolist() == [1.234]
    assert component_sum == 1.234
    assert filter_text == "#1179"


def test_assemble_component_analysis_can_pick_inverter_ids() -> None:
    """Analysis should select inverter-labelled columns when requested."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "PV #1133 - PV Meter": [-2.0],
            "PV #1134 - PV Inverter": [-1.0],
        }
    )
    mcfg = _DummyMicrogridConfig(
        {"pv": _DummyComponentConfig(meter=[1133], inverter=[1134])}
    )

    analyse_df, component_sum, _ = assemble_component_analysis(
        component_filter=["All"],
        component_key="pv",
        component_types=["pv"],
        energy_report_df=energy_report_df,
        timestep_hours=1.0,
        mapper=ColumnMapper.from_default(locale="en"),
        component_label="PV",
        value_col_name="pv_asset_production",
        invert_sign=True,
        trunc_values=True,
        mcfg=cast(MicrogridConfig, mcfg),
        component_id_source="inverter",
    )

    assert analyse_df["PV"].tolist() == ["PV #1134 - PV Inverter"]
    assert analyse_df["PV-Production"].tolist() == [1.0]
    assert component_sum == 1.0


def test_assemble_component_analysis_can_pick_meter_ids() -> None:
    """Analysis should select meter-labelled columns when requested."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "Battery #1532 - Battery Meter": [5.0],
            "Battery #1533 - Battery Inverter": [4.0],
        }
    )
    mcfg = _DummyMicrogridConfig(
        {"battery": _DummyComponentConfig(meter=[1532], inverter=[1533])}
    )

    analyse_df, component_sum, _ = assemble_component_analysis(
        component_filter=["All"],
        component_key="battery",
        component_types=["battery"],
        energy_report_df=energy_report_df,
        timestep_hours=1.0,
        mapper=ColumnMapper.from_default(locale="en"),
        component_label="Battery",
        value_col_name="battery_power_flow",
        mcfg=cast(MicrogridConfig, mcfg),
        component_id_source="meter",
    )

    assert analyse_df["Battery"].tolist() == ["Battery #1532 - Battery Meter"]
    assert analyse_df["Battery Power Flow"].tolist() == [5.0]
    assert component_sum == 5.0


def test_assemble_component_analysis_returns_empty_when_id_source_has_no_matches() -> (
    None
):
    """Selecting an id source with no matching columns should return empty output."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "PV #1134 - PV Inverter": [-1.0],
        }
    )
    mcfg = _DummyMicrogridConfig({"pv": _DummyComponentConfig(meter=[1133])})

    analyse_df, component_sum, filter_text = assemble_component_analysis(
        component_filter=["All"],
        component_key="pv",
        component_types=["pv"],
        energy_report_df=energy_report_df,
        timestep_hours=1.0,
        mapper=ColumnMapper.from_default(locale="en"),
        component_label="PV",
        value_col_name="pv_asset_production",
        mcfg=cast(MicrogridConfig, mcfg),
        component_id_source="meter",
    )

    assert analyse_df.empty
    assert component_sum == 0
    assert filter_text == "All"


def test_build_overview_df_keeps_expected_optional_columns() -> None:
    """Overview output should preserve order and ignore unavailable columns."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "grid_consumption": [3.0],
            "mid_consumption": [7.0],
            "grid_feed_in": [1.0],
            "pv_asset_production": [5.0],
            "extra_column": [99.0],
        }
    )

    result = build_overview_df(energy_report_df, component_types=["pv", "wind"])

    expected = energy_report_df[
        [
            "timestamp",
            "grid_consumption",
            "mid_consumption",
            "grid_feed_in",
            "pv_asset_production",
        ]
    ].rename(columns={"pv_asset_production": "pv"})

    assert_frame_equal(result, expected)


def test_build_overview_df_adds_battery_helpers() -> None:
    """Battery output keeps requested columns and adds derived helpers."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "battery_soc_pct": [42.0, 53.0],
            "pv_asset_production": [12.0, 10.0],
            "chp_asset_production": [2.0, 1.0],
            "wind_asset_production": [3.0, 4.0],
        }
    )
    result = build_overview_df(
        energy_report_df,
        component_types=["pv", "chp", "wind", "battery"],
    )

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "grid_consumption": [30.0, 25.0],
            "mid_consumption": [35.0, 21.0],
            "pv": [12.0, 10.0],
            "chp": [2.0, 1.0],
            "wind": [3.0, 4.0],
            "battery_power_flow": [5.0, -4.0],
            "battery_soc_pct": [42.0, 53.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_charge": [5.0, 0.0],
            "battery_discharge": [0.0, -4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_build_overview_df_battery_requires_configured_columns() -> None:
    """Battery output should fail clearly when required columns are missing."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "mid_consumption": [35.0],
            "grid_consumption": [30.0],
        }
    )

    with pytest.raises(KeyError, match="battery_power_flow"):
        build_overview_df(
            energy_report_df,
            component_types=["battery"],
        )


def test_compute_energy_summary_includes_rollups_and_percentages() -> None:
    """Energy summaries should include rollups and stable ordering."""
    df = pd.DataFrame(
        {
            "pv_asset_production": [4.0, 4.0],
            "chp_asset_production": [1.0, 1.0],
            "grid_consumption": [2.0, 0.0],
        }
    )

    result = compute_energy_summary(
        df,
        resolution=timedelta(minutes=30),
        include_rollups=True,
    )

    expected = pd.DataFrame(
        {
            "Energy Source": [
                "PV",
                "CHP",
                "Production (PV+Wind+CHP)",
                "Grid Consumption",
            ],
            "Energy [kWh]": [4.0, 1.0, 5.0, 1.0],
            "Power [kW]": [8.0, 2.0, 10.0, 2.0],
            "Mean [kW]": [4.0, 1.0, 5.0, 1.0],
            "Energy %": [36.364, 9.091, 45.455, 9.091],
        }
    )

    assert_frame_equal(result, expected)


def test_compute_energy_summary_uses_positive_grid_consumption_only() -> None:
    """Energy summaries should not let grid export cancel grid import."""
    df = pd.DataFrame(
        {
            "pv_asset_production": [2.0, 0.0],
            "grid_consumption": [-5.0, 3.0],
        }
    )

    result = compute_energy_summary(
        df,
        resolution=timedelta(hours=1),
    )

    expected = pd.DataFrame(
        {
            "Energy Source": ["PV", "Grid Consumption"],
            "Energy [kWh]": [2.0, 3.0],
            "Power [kW]": [2.0, 3.0],
            "Mean [kW]": [1.0, 1.5],
            "Energy %": [40.0, 60.0],
        }
    )

    assert_frame_equal(result, expected)


def test_compute_energy_summary_rejects_non_positive_resolution() -> None:
    """A non-positive aggregation step must fail clearly."""
    with pytest.raises(ValueError, match="resolution must be positive"):
        compute_energy_summary(pd.DataFrame({"grid_consumption": [1.0]}), timedelta(0))


def test_aggregate_metrics_computes_energy_peak_date_and_pricing() -> None:
    """Aggregate metrics should clip import, localize peak dates, and price energy."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-14 22:00:00", "2026-07-15 00:00:00"]),
            "pv_asset_production": [1.0, 2.0],
            "chp_asset_production": [0.0, 1.0],
            "production_self_use": [0.5, 1.0],
            "grid_feed_in": [0.0, 4.0],
            "grid_consumption": [-1.0, 5.0],
            "mid_consumption": [2.0, 6.0],
            "day_ahead_price": [100.0, 200.0],
        }
    )

    result = aggregate_metrics(
        energy_report_df,
        resolution=timedelta(hours=1),
        price_column="day_ahead_price",
    )

    assert result == {
        "pv_production_sum": 3.0,
        "chp_production_sum": 1.0,
        "wind_production_sum": 0.0,
        "prod_self_consumption_sum": 1.5,
        "grid_feed_in_sum": 4.0,
        "grid_consumption_sum": 5.0,
        "mid_consumption_sum": 8.0,
        "total_production_sum": 4.0,
        "production_to_battery_sum": 0.0,
        "prod_bat_sum": 0.0,
        "grid_to_battery_sum": 0.0,
        "battery_to_grid_sum": 0.0,
        "battery_to_consumption_sum": 0.0,
        "prod_self_consumption_share": 0.1875,
        "prod_self_production_share": 0.375,
        "peak": 5.0,
        "peak_date": "15.07.2026",
        "grid_import_cost_sum": 1.0,
        "grid_feed_in_revenue_sum": 0.8,
    }


def test_aggregate_metrics_adds_battery_flow_splits() -> None:
    """Battery charge/discharge should be split by production, grid, and load."""
    energy_report_df = pd.DataFrame(
        {
            "pv_asset_production": [10.0, 2.0, 1.0, 8.0],
            "mid_consumption": [4.0, 5.0, 4.0, 3.0],
            "battery_power_flow": [3.0, 4.0, -2.0, -2.0],
            "production_excess": [6.0, 0.0, 0.0, 5.0],
            "grid_feed_in": [3.0, 0.0, 0.0, 7.0],
            "grid_consumption": [-3.0, 7.0, 1.0, -7.0],
            "production_to_battery": [3.0, 0.0, 0.0, 0.0],
            "grid_to_battery": [0.0, 4.0, 0.0, 0.0],
            "battery_to_grid": [0.0, 0.0, 0.0, 2.0],
            "battery_to_consumption": [0.0, 0.0, 2.0, 0.0],
        }
    )

    result = aggregate_metrics(
        energy_report_df,
        resolution=timedelta(hours=1),
    )

    assert result["production_to_battery_sum"] == 3.0
    assert result["prod_bat_sum"] == 3.0
    assert result["grid_to_battery_sum"] == 4.0
    assert result["battery_to_grid_sum"] == 2.0
    assert result["battery_to_consumption_sum"] == 2.0
    assert result["grid_consumption_sum"] == 8.0


def test_aggregate_metrics_aggregates_production_to_battery() -> None:
    """production_to_battery_sum is aggregated from the upstream flow column."""
    energy_report_df = pd.DataFrame(
        {
            "pv_asset_production": [10.0],
            "mid_consumption": [4.0],
            "battery_power_flow": [1.0],
            "grid_feed_in": [5.0],
            "production_to_battery": [1.0],
            "grid_to_battery": [0.0],
        }
    )

    result = aggregate_metrics(
        energy_report_df,
        resolution=timedelta(hours=1),
    )

    assert result["production_to_battery_sum"] == 1.0
    assert result["grid_to_battery_sum"] == 0.0
