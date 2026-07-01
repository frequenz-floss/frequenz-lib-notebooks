# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting plotting helpers."""

from __future__ import annotations

import sys
import types

import pandas as pd

from frequenz.lib.notebooks.reporting.plotter import (  # noqa: E402
    plot_time_series_battery_usecase,
)

gridpool = sys.modules.setdefault(
    "frequenz.gridpool", types.ModuleType("frequenz.gridpool")
)

if not hasattr(gridpool, "MicrogridConfig"):
    setattr(gridpool, "MicrogridConfig", object)


def test_plot_time_series_battery_usecase_adds_peak_lines() -> None:
    """Peak reference lines should be plotted even when omitted from cols."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "grid_consumption_without_battery": [35.0, 21.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [0.0, -4.0],
            "pv": [12.0, 10.0],
        }
    )

    fig = plot_time_series_battery_usecase(
        df,
        time_col="timestamp",
        cols=[
            "grid_consumption",
            "grid_consumption_without_battery",
            "battery_discharge",
            "battery_charge",
            "pv",
        ],
    )

    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert "Lastspitze vor optimierung" in traces_by_name
    assert "Lastspitze nach optimierung" in traces_by_name
    assert traces_by_name["Lastspitze vor optimierung"].line.dash == "dot"
    assert traces_by_name["Lastspitze nach optimierung"].line.dash == "dot"


def test_plot_time_series_battery_usecase_accepts_legacy_german_columns() -> None:
    """Legacy German column names should still render through the adapter layer."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "Netzbezug": [30.0, 25.0],
            "Batterie Leistungsfluss": [5.0, -4.0],
            "Netzbezug ohne Batterie": [35.0, 21.0],
            "Lastspitze vor optimierung": [35.0, 35.0],
            "Lastspitze nach optimierung": [30.0, 30.0],
            "Batterie Entladung": [5.0, 0.0],
            "Batterie Beladung": [0.0, -4.0],
            "PV": [12.0, 10.0],
        }
    )

    fig = plot_time_series_battery_usecase(df, time_col="timestamp")
    trace_names = [trace.name for trace in fig.data if getattr(trace, "name", None)]

    assert "Netzbezug" in trace_names
    assert "Lastspitze vor optimierung" in trace_names
    assert "Lastspitze nach optimierung" in trace_names
