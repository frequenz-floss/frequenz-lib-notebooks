# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for the reporting helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import cast

import pandas as pd
import pytest
import pytz
from frequenz.gridpool import MicrogridConfig
from pandas.testing import assert_frame_equal, assert_series_equal

from frequenz.lib.notebooks.reporting.utils.helpers import (
    _column_has_data,
    _get_numeric_series,
    _sum_cols,
    build_color_map,
    convert_timezone,
    fmt_to_de_system,
    label_component_columns,
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

    def component_types(self) -> list[str]:
        return list(self.mapping.keys())

    def component_type_ids(self, component_type: str) -> list[str]:
        return self.mapping.get(component_type, [])


def test_label_component_columns_applies_expected_prefixes() -> None:
    """Numeric columns are renamed with component labels while others stay untouched."""
    df = pd.DataFrame(
        {
            "1": [10],
            "2": [20],
            "3": [30],
            "4": [40],
            "constant": [99],
        }
    )
    config = _DummyMicrogridConfig(
        {"battery": ["1"], "pv": ["2"], "ev": ["3"], "chp": ["4"]}
    )

    renamed, labels = label_component_columns(
        df,
        cast(MicrogridConfig, config),
    )

    assert renamed.columns.tolist() == [
        "Battery #1",
        "PV #2",
        "EV #3",
        "CHP #4",
        "constant",
    ]
    assert labels == ["Battery #1", "PV #2", "EV #3", "CHP #4"]


def test_set_date_to_midnight_creates_timezone_aware_midnight() -> None:
    """Date and datetime inputs both produce midnight timestamps in the target TZ."""
    result_date = set_date_to_midnight(date(2024, 5, 1), "Europe/Berlin")
    expected_date = pytz.timezone("Europe/Berlin").localize(datetime(2024, 5, 1))
    assert result_date == expected_date

    noon_input = datetime(2024, 5, 2, 12, 30)
    result_datetime = set_date_to_midnight(noon_input, "UTC")
    tz_utc = pytz.timezone("UTC")
    assert result_datetime == tz_utc.localize(datetime(2024, 5, 2))


def test_set_date_to_midnight_unknown_timezone_warns_and_falls_back() -> None:
    """Unknown timezone name triggers a warning and falls back to UTC."""
    with pytest.warns(RuntimeWarning):
        result = set_date_to_midnight(date(2024, 6, 1), "Invalid/Zone")
    assert result == pytz.timezone("UTC").localize(datetime(2024, 6, 1))


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
