# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for the reporting helper utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import cast
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from frequenz.client.common.microgrid import MicrogridId
from pandas.testing import assert_frame_equal, assert_series_equal

from frequenz.lib.notebooks.reporting.utils.helpers import (
    _column_has_data,
    _get_numeric_series,
    _sum_cols,
    build_color_map,
    convert_timezone,
    fill_aggregated_component_columns,
    fmt_to_de_system,
    get_meter_display_names,
    long_to_wide,
    set_date_to_midnight,
)


def test_get_numeric_series_handles_missing_columns() -> None:
    """Series is zero-filled when column is missing or None."""
    df = pd.DataFrame({"a": [1, "2", None]}, index=pd.RangeIndex(3))

    result_none = _get_numeric_series(df, None)
    expected_zero = pd.Series(0.0, index=df.index, dtype="float64", name=None)
    assert_series_equal(result_none, expected_zero)

    result_missing = _get_numeric_series(df, "missing")
    expected_missing = expected_zero.copy()
    expected_missing.name = "missing"
    assert_series_equal(result_missing, expected_missing)

    result_cast = _get_numeric_series(df, "a")
    expected_cast = pd.Series(
        [1.0, 2.0, 0.0], index=df.index, dtype="float64", name="a"
    )
    assert_series_equal(result_cast, expected_cast)


def test_sum_cols_handles_empty_and_missing_inputs() -> None:
    """Summation skips missing columns and returns zeros for empty input."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=pd.RangeIndex(2))

    summed = _sum_cols(df, ["a", "b", "missing"])
    expected = pd.Series([4.0, 6.0], index=df.index, dtype="float64")
    assert_series_equal(summed, expected)

    empty = _sum_cols(df, [])
    assert_series_equal(empty, pd.Series(0.0, index=df.index, dtype="float64"))


def test_column_has_data_checks_numeric_content() -> None:
    """Boolean helper only returns True for existing columns with non-zero data."""
    df = pd.DataFrame(
        {
            "non_zero": [0, 1, 0],
            "zeros": [0, 0, 0],
            "nans": [float("nan"), float("nan"), 0],
        }
    )

    assert _column_has_data(df, "non_zero")
    assert not _column_has_data(df, "zeros")
    assert not _column_has_data(df, "nans")
    assert not _column_has_data(df, "missing")


def test_fmt_to_de_system_formats_with_german_conventions() -> None:
    """Number formatting uses comma decimal separator and dot thousands separator."""
    assert fmt_to_de_system(12345.678) == "12.345,68"
    assert fmt_to_de_system(-9876.0) == "-9.876,00"


def test_convert_timezone_localizes_and_converts_series() -> None:
    """Naive datetimes are localized before conversion to the target timezone."""
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=2, freq="h"))
    converted = convert_timezone(
        timestamps,
        target_tz="Europe/Berlin",
        assume_tz="UTC",
    )
    expected = timestamps.dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
    assert_series_equal(converted, expected)


def test_convert_timezone_aware_series_and_type_validation() -> None:
    """Timezone-aware Series is converted while invalid input raises ValueError."""
    aware = pd.Series(pd.date_range("2024-02-01", periods=2, freq="h", tz="UTC"))
    converted = convert_timezone(aware, target_tz="Europe/Berlin")
    expected = aware.dt.tz_convert("Europe/Berlin")
    assert_series_equal(converted, expected)

    bad_input = cast(pd.Series, pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError):
        convert_timezone(bad_input)


@dataclass
class _DummyMicrogridConfig:
    """Minimal config stub exposing component type helpers."""

    mapping: dict[str, list[str]]
    ctype: dict[str, object] | None = None

    def component_types(self) -> list[str]:
        if self.ctype:
            return list(self.ctype.keys())
        return list(self.mapping.keys())

    def component_type_ids(self, component_type: str) -> list[str]:
        return self.mapping.get(component_type, [])


@dataclass
class _DummyComponentConfig:
    """Minimal component config stub with meter/inverter/component groups."""

    meter: list[str] | None = None
    inverter: list[str] | None = None
    component: list[str] | None = None


def test_get_meter_display_names_runs_inside_active_event_loop() -> None:
    """The sync helper should work even when an event loop is already active."""

    class _Category:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Component:
        def __init__(self, component_id: str, name: str, category: str) -> None:
            self.id = component_id
            self.name = name
            self.category = _Category(category)

    class _FakeClient:
        async def list_microgrid_electrical_components(
            self, microgrid_id: MicrogridId
        ) -> list[_Component]:
            assert microgrid_id == MicrogridId(241)
            return [
                _Component("ElectricalComponentId(1179)", "meter_pq_0", "METER"),
                _Component("ElectricalComponentId(1188)", "meter_GT", "METER"),
                _Component("ElectricalComponentId(1189)", "GT_1", "CHP"),
            ]

    async def _run_test() -> None:
        result = get_meter_display_names(
            241, server_url="grpc://assets.example.com:443"
        )
        assert result == {"1179": "meter_pq_0", "1188": "meter_GT"}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "frequenz.lib.notebooks.reporting.utils.helpers.AssetsApiClient",
            lambda server_url, auth_key=None, sign_secret=None: _FakeClient(),
        )
        asyncio.run(_run_test())


def test_set_date_to_midnight_creates_timezone_aware_midnight() -> None:
    """Date and datetime inputs both produce midnight timestamps in the target TZ."""
    result_date = set_date_to_midnight(date(2024, 5, 1), "Europe/Berlin")
    expected_date = datetime(2024, 5, 1, tzinfo=ZoneInfo("Europe/Berlin"))
    assert result_date == expected_date

    noon_input = datetime(2024, 5, 2, 12, 30, tzinfo=UTC)
    result_datetime = set_date_to_midnight(noon_input, "UTC")
    assert result_datetime == datetime(2024, 5, 2, tzinfo=ZoneInfo("UTC"))


def test_set_date_to_midnight_unknown_timezone_warns_and_falls_back() -> None:
    """Unknown timezone name triggers a warning and falls back to UTC."""
    with pytest.warns(RuntimeWarning):
        result = set_date_to_midnight(date(2024, 6, 1), "Invalid/Zone")
    assert result == datetime(2024, 6, 1, tzinfo=ZoneInfo("UTC"))


def test_long_to_wide_pivots_and_adds_sum_column() -> None:
    """Long-format table is pivoted and an aggregate column is appended."""
    df = pd.DataFrame(
        {
            "Timestamp": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
            ],
            "Battery": ["A", "B", "A", "B"],
            "Battery Throughput": [1.0, 2.0, 3.0, 4.0],
        }
    )

    wide = long_to_wide(df)
    expected = pd.DataFrame(
        {
            "A": [1.0, 3.0],
            "B": [2.0, 4.0],
            "Battery Throughput Sum": [3.0, 7.0],
        },
        index=pd.Index(
            [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            name="Timestamp",
        ),
    )
    assert_frame_equal(wide, expected)


def test_fill_aggregated_component_columns_backfills_missing_summaries() -> None:
    """Aggregate columns are filled only where NaNs exist via per-component sums."""
    df = pd.DataFrame(
        {
            "battery_power_flow": [0.0, float("nan"), 5.0],
            "Battery #1": [1.0, 2.0, 2.5],
            "Battery #2": [3.0, 4.0, 2.5],
            "pv_asset_production": [float("nan"), float("nan"), 1.0],
            "PV #1": [0.5, 0.75, 0.3],
            "PV #2": [0.5, 0.25, 0.0],
        }
    )

    result = fill_aggregated_component_columns(
        df.copy(), component_types=["battery", "pv"]
    )

    assert result["battery_power_flow"].tolist() == [0.0, 6.0, 5.0]
    assert result["pv_asset_production"].tolist() == [1.0, 1.0, 1.0]


def test_fill_aggregated_component_columns_respects_component_filter() -> None:
    """Only requested component types trigger aggregation fill-ins."""
    df = pd.DataFrame(
        {
            "battery_power_flow": [0.0, float("nan")],
            "Battery #1": [1.0, 2.0],
            "Battery #2": [3.0, 4.0],
            "pv_asset_production": [float("nan"), float("nan")],
            "PV #1": [0.5, 0.75],
            "PV #2": [0.5, 0.25],
        }
    )

    result = fill_aggregated_component_columns(df.copy(), component_types=["battery"])

    assert result["pv_asset_production"].iloc[:2].isna().all()


def test_fill_aggregated_component_columns_accepts_custom_config() -> None:
    """An external config mapping can drive aggregation of bespoke components."""
    df = pd.DataFrame(
        {
            "custom_power": [float("nan"), 4.0],
            "Custom #1": [1.0, 2.0],
            "Custom #2": [2.0, 3.0],
        }
    )

    config = {"custom": ("custom_power", "Custom #")}

    result = fill_aggregated_component_columns(
        df.copy(),
        component_types=["custom"],
        config=config,
    )

    assert result["custom_power"].tolist() == [3.0, 4.0]


def test_fill_aggregated_component_columns_accepts_explicit_component_columns() -> None:
    """Explicit canonical component columns can drive aggregation fill-ins."""
    df = pd.DataFrame(
        {
            "battery_power_flow": [float("nan"), 4.0],
            "1532": [1.0, 2.0],
            "1533": [2.0, 3.0],
        }
    )

    result = fill_aggregated_component_columns(
        df.copy(),
        component_types=["battery"],
        component_columns_by_type={"battery": ["1532", "1533"]},
    )

    assert result["battery_power_flow"].tolist() == [3.0, 4.0]


def test_fill_aggregated_component_columns_falls_back_to_legacy_prefixes() -> None:
    """Legacy prefixed columns should still fill aggregates when explicit ids miss."""
    df = pd.DataFrame(
        {
            "battery_power_flow": [float("nan"), 4.0],
            "Battery #1532": [1.0, 2.0],
            "Battery #1533": [2.0, 3.0],
        }
    )

    result = fill_aggregated_component_columns(
        df.copy(),
        component_types=["battery"],
        component_columns_by_type={"battery": []},
    )

    assert result["battery_power_flow"].tolist() == [3.0, 4.0]


def test_long_to_wide_custom_sum_column_name() -> None:
    """Custom sum column name is respected when provided."""
    df = pd.DataFrame(
        {
            "Timestamp": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "Battery": ["A", "A"],
            "Battery Throughput": [1.0, 2.0],
        }
    )

    wide = long_to_wide(df, sum_col_name="total")
    assert "total" in wide.columns
    assert wide.loc[pd.Timestamp("2024-01-01"), "total"] == 3.0


def test_build_color_map_respects_user_colors_and_palette() -> None:
    """Color map uses user-provided colors before assigning palette entries."""
    cols = ["alpha", "beta", "gamma"]
    color_dict = {"alpha": "#ff0000"}
    palette = ["#00ff00", "#0000ff"]

    color_map = build_color_map(cols, color_dict=color_dict, palette=palette)

    assert color_map["alpha"] == "rgba(255,0,0,1.000)"
    assert color_map["beta"] == "rgba(0,255,0,1.000)"
    assert color_map["gamma"] == "rgba(0,0,255,1.000)"
