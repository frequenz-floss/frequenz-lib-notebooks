# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting notebook utility functions."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper
from frequenz.lib.notebooks.reporting.utils.component_metadata import (
    ComponentMetadata,
)
from frequenz.lib.notebooks.reporting.utils.reporting_nb_functions import (
    aggregate_metrics,
    assemble_component_analysis,
    build_component_analysis,
    build_overview_df,
    compute_energy_summary,
    resolve_component_analysis_ids,
)


class _DummyComponentConfig:
    """Minimal component config stub with id groups."""

    def __init__(
        self,
        *,
        meter: list[int] | None = None,
        inverter: list[int] | None = None,
        component: list[int] | None = None,
    ) -> None:
        self.meter = meter
        self.inverter = inverter
        self.component = component


class _DummyMicrogridConfig:
    """Minimal microgrid config stub exposing the ctype mapping."""

    def __init__(self, ctype: dict[str, _DummyComponentConfig]) -> None:
        self.ctype = ctype


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


def test_build_component_analysis_uses_raw_component_ids_with_display_metadata() -> (
    None
):
    """Raw component ID columns should render meter names from dataframe metadata."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-09 06:30:00", "2026-01-09 07:00:00"], utc=True
            ),
            "1179": [1.2, 1.5],
            "1188": [0.7, 0.8],
        }
    )
    component_metadata = ComponentMetadata(
        display_names={
            "1179": "PV Roof Meter",
            "1188": "PV Yard Meter",
        },
        ids_by_type={"pv": ["1179", "1188"]},
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["All"],
        component_label="PV",
        value_col_name="pv_asset_production",
        component_metadata=component_metadata,
    )

    assert result["PV"].tolist() == [
        "PV Roof Meter",
        "PV Roof Meter",
        "PV Yard Meter",
        "PV Yard Meter",
    ]
    assert result["pv_asset_production"].tolist() == [1.2, 1.5, 0.7, 0.8]


def test_build_component_analysis_keeps_id_when_no_display_name_is_present() -> None:
    """Legacy `#id` labels should still work when no meter name is available."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "Battery #1532": [5.0],
        }
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["#1532"],
        component_label="Battery",
        value_col_name="battery_power_flow",
    )

    assert result["Battery"].tolist() == ["#1532"]


def test_build_component_analysis_can_restrict_to_allowed_component_ids() -> None:
    """Allowed ids should constrain the selected columns for `All` filters."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "PV #1133 PV Meter": [2.0],
            "PV #1134 PV Inverter": [1.0],
        }
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["All"],
        component_label="PV",
        value_col_name="pv_asset_production",
        allowed_component_ids={"1134"},
    )

    assert result["PV"].tolist() == ["PV Inverter"]
    assert result["pv_asset_production"].tolist() == [1.0]


def test_build_component_analysis_can_reselect_legacy_component_by_display_value() -> (
    None
):
    """Legacy labels returned by the formatter should round-trip through selection."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "PV #1134 PV Inverter": [1.0],
        }
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["PV Inverter"],
        component_label="PV",
        value_col_name="pv_asset_production",
    )

    assert result["PV"].tolist() == ["PV Inverter"]
    assert result["pv_asset_production"].tolist() == [1.0]


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


def test_resolve_component_analysis_ids_supports_meter_and_inverter() -> None:
    """The helper should expose the requested id group from the config."""
    mcfg = _DummyMicrogridConfig(
        {
            "pv": _DummyComponentConfig(meter=[1133, 1141], inverter=[1134, 1135]),
        }
    )

    assert resolve_component_analysis_ids(mcfg, "pv", "meter") == {"1133", "1141"}
    assert resolve_component_analysis_ids(mcfg, "pv", "inverter") == {
        "1134",
        "1135",
    }
    assert resolve_component_analysis_ids(mcfg, "pv", "all") is None


def test_resolve_component_analysis_ids_returns_empty_set_for_missing_group() -> None:
    """A missing id group should produce an empty allowed-id set."""
    mcfg = _DummyMicrogridConfig({"pv": _DummyComponentConfig(meter=[1133])})

    assert resolve_component_analysis_ids(mcfg, "pv", "component") == set()


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
                "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
                "PV #1179 PV Roof Meter": ["-1.2345"],
            }
        ),
        timestep_hours=1.0,
        mapper=ColumnMapper.from_default(locale="en"),
        component_label="PV",
        value_col_name="pv_asset_production",
        invert_sign=True,
        trunc_values=True,
    )

    assert analyse_df["PV"].tolist() == ["PV Roof Meter"]
    assert analyse_df["PV-Production"].tolist() == [1.234]
    assert component_sum == 1.234
    assert filter_text == "#1179"


def test_assemble_component_analysis_can_pick_inverter_ids() -> None:
    """Analysis should select inverter-labelled columns when requested."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "PV #1133 PV Meter": [-2.0],
            "PV #1134 PV Inverter": [-1.0],
        }
    )
    mcfg = _DummyMicrogridConfig(
        {
            "pv": _DummyComponentConfig(meter=[1133], inverter=[1134]),
        }
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
        mcfg=mcfg,
        component_id_source="inverter",
    )

    assert analyse_df["PV"].tolist() == ["PV Inverter"]
    assert analyse_df["PV-Production"].tolist() == [1.0]
    assert component_sum == 1.0


def test_assemble_component_analysis_can_pick_meter_ids() -> None:
    """Analysis should select meter-labelled columns when requested."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "Battery #1532 Battery Meter": [5.0],
            "Battery #1533 Battery Inverter": [4.0],
        }
    )
    mcfg = _DummyMicrogridConfig(
        {
            "battery": _DummyComponentConfig(meter=[1532], inverter=[1533]),
        }
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
        mcfg=mcfg,
        component_id_source="meter",
    )

    assert analyse_df["Battery"].tolist() == ["Battery Meter"]
    assert analyse_df["Battery Power Flow"].tolist() == [5.0]
    assert component_sum == 5.0


def test_assemble_component_analysis_returns_empty_when_id_source_has_no_matches() -> (
    None
):
    """Selecting an id source with no matching columns should return empty output."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "PV #1134 PV Inverter": [-1.0],
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
        mcfg=mcfg,
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
            "battery_power_flow": [-2.0],
            "extra_column": [99.0],
        }
    )

    result = build_overview_df(energy_report_df, component_types=["pv", "battery"])

    expected = energy_report_df[
        [
            "timestamp",
            "grid_consumption",
            "mid_consumption",
            "grid_feed_in",
            "pv_asset_production",
            "battery_power_flow",
        ]
    ]

    assert_frame_equal(result, expected)


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
            "production_excess_in_bat": [0.2, 0.3],
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
        "prod_bat_sum": 0.5,
        "grid_feed_in_sum": 4.0,
        "grid_consumption_sum": 5.0,
        "mid_consumption_sum": 8.0,
        "total_production_sum": 4.0,
        "prod_self_consumption_share": 0.1875,
        "prod_self_production_share": 0.375,
        "peak": 5.0,
        "peak_date": "15.07.2026",
        "grid_import_cost_sum": 1.0,
        "grid_feed_in_revenue_sum": 0.8,
    }
