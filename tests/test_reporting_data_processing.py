# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting data preparation helpers."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from frequenz.lib.notebooks.reporting.data_processing import create_battery_usecase_df


def test_create_battery_usecase_df_builds_expected_columns() -> None:
    """Battery usecase helper derives the expected canonical columns."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
        }
    )
    result = create_battery_usecase_df(energy_report_df)

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "grid_consumption_without_battery": [35.0, 21.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [0.0, -4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_create_battery_usecase_df_preserves_pv_when_available() -> None:
    """Optional PV production is retained in the standardized output."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv_asset_production": [12.0, 10.0],
        }
    )
    result = create_battery_usecase_df(energy_report_df)

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv": [12.0, 10.0],
            "grid_consumption_without_battery": [35.0, 21.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [0.0, -4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_create_battery_usecase_df_accepts_custom_input_column_names() -> None:
    """Custom source column names are normalized to the canonical output schema."""
    energy_report_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "grid_load": [30.0, 25.0],
            "battery_flow": [5.0, -4.0],
            "pv_power": [12.0, 10.0],
        }
    )
    result = create_battery_usecase_df(
        energy_report_df,
        timestamp_col="time",
        grid_consumption_col="grid_load",
        battery_col="battery_flow",
        pv_col="pv_power",
    )

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv": [12.0, 10.0],
            "grid_consumption_without_battery": [35.0, 21.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [0.0, -4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_create_battery_usecase_df_requires_configured_input_columns() -> None:
    """Missing required columns should fail clearly."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "grid_consumption": [30.0],
        }
    )

    with pytest.raises(KeyError, match="battery_power_flow"):
        create_battery_usecase_df(energy_report_df)
