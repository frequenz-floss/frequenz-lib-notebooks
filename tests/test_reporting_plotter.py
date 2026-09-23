# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting plotting helpers."""

from __future__ import annotations

import sys
import types
import warnings
from math import isclose

import pandas as pd

from frequenz.lib.notebooks.reporting.plotter import (  # noqa: E402
    plot_time_series_battery_soc,
    plot_time_series_battery_soc_and_usecase,
    plot_time_series_battery_usecase,
    summary_plot,
)
from frequenz.lib.notebooks.reporting.utils.colors import COLOR_DICT
from frequenz.lib.notebooks.reporting.utils.helpers import SummaryPeriod

gridpool = sys.modules.setdefault(
    "frequenz.gridpool", types.ModuleType("frequenz.gridpool")
)

if not hasattr(gridpool, "MicrogridConfig"):
    setattr(gridpool, "MicrogridConfig", object)


# pylint:disable=too-many-statements
def test_summary_plot_aggregates_standard_columns_to_mwh() -> None:
    """Standard reporting columns should aggregate by month in MWh."""
    df = pd.DataFrame(
        {
            "grid_consumption": [1000.0, 2000.0],
            "grid_feed_in": [500.0, 1000.0],
            "battery_charge": [250.0, 250.0],
            "battery_discharge": [100.0, 200.0],
            "mid_consumption": [3000.0, 1000.0],
            "pv_asset_production": [2000.0, 2000.0],
            "chp_asset_production": [1000.0, 1000.0],
            "wind_asset_production": [500.0, 500.0],
            "idle": [0.0, 0.0],
        },
        index=pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"]),
    )

    fig, months = summary_plot(df, period="monthly")
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert months.index.tolist() == [pd.Timestamp("2026-01-01").date()]
    assert all("_pos" not in col and "_neg" not in col for col in months.columns)
    assert isclose(months.loc[months.index[0], "Netzbezug"], 3.0)
    assert isclose(months.loc[months.index[0], "Netz Einspeisung"], -1.5)
    assert isclose(months.loc[months.index[0], "PV-Erzeugung"], -4.0)
    assert isclose(months.loc[months.index[0], "BHKW-Erzeugung"], -2.0)
    assert isclose(months.loc[months.index[0], "Wind-Erzeugung"], -1.0)
    assert "Netzbezug" in traces_by_name
    assert "Netz Einspeisung" in traces_by_name
    assert "Batterie Beladung" in traces_by_name
    assert "Batterie Entladung" in traces_by_name
    assert "MID Gesamtverbrauch" in traces_by_name
    assert "PV-Erzeugung" in traces_by_name
    assert "BHKW-Erzeugung" in traces_by_name
    assert "Wind-Erzeugung" in traces_by_name
    assert "CHP" not in traces_by_name
    assert "PV" not in traces_by_name
    assert "Wind" not in traces_by_name
    assert "idle" not in traces_by_name
    assert list(traces_by_name["Netzbezug"].y) == [3.0]
    assert list(traces_by_name["Netz Einspeisung"].y) == [-1.5]
    assert list(traces_by_name["Batterie Beladung"].y) == [0.5]
    assert list(traces_by_name["Batterie Entladung"].y) == [-0.3]
    assert list(traces_by_name["PV-Erzeugung"].y) == [-4.0]
    assert list(traces_by_name["BHKW-Erzeugung"].y) == [-2.0]
    assert list(traces_by_name["Wind-Erzeugung"].y) == [-1.0]
    assert traces_by_name["Netzbezug"].marker.color == COLOR_DICT["Netzbezug"]
    assert traces_by_name["Netz Einspeisung"].marker.color == COLOR_DICT["Netzbezug"]
    assert traces_by_name["PV-Erzeugung"].marker.color == COLOR_DICT["PV-Erzeugung"]
    assert (
        traces_by_name["Batterie Beladung"].marker.color
        == COLOR_DICT["Batterie Entladung"]
    )
    assert (
        traces_by_name["Batterie Entladung"].marker.color
        == COLOR_DICT["Batterie Entladung"]
    )
    assert traces_by_name["MID Gesamtverbrauch"].marker.color == "rgba(121, 85, 72, 1)"
    assert traces_by_name["BHKW-Erzeugung"].marker.color == COLOR_DICT["BHKW-Erzeugung"]
    assert traces_by_name["Wind-Erzeugung"].marker.color == COLOR_DICT["Wind-Erzeugung"]
    assert traces_by_name["Netzbezug"].opacity is None
    assert traces_by_name["MID Gesamtverbrauch"].opacity is None
    assert traces_by_name["Batterie Beladung"].opacity is None
    assert traces_by_name["Netz Einspeisung"].opacity == 0.75
    assert traces_by_name["Batterie Entladung"].opacity == 0.75
    assert traces_by_name["PV-Erzeugung"].opacity == 0.75
    assert traces_by_name["BHKW-Erzeugung"].opacity == 0.75
    assert traces_by_name["Wind-Erzeugung"].opacity == 0.75
    assert fig.layout.barmode == "relative"
    assert {
        trace.name: trace.offsetgroup
        for trace in fig.data
        if getattr(trace, "name", None)
    } == {
        "Netzbezug": "grid",
        "Netz Einspeisung": "grid",
        "MID Gesamtverbrauch": "consumption",
        "Batterie Beladung": "battery",
        "Batterie Entladung": "battery",
        "PV-Erzeugung": "production",
        "BHKW-Erzeugung": "production",
        "Wind-Erzeugung": "production",
    }
    assert all(trace.yaxis is None for trace in fig.data)
    assert fig.layout.title.y == 0.98
    assert fig.layout.title.pad.t == 0
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.legend.x == 0
    assert fig.layout.legend.y == 1.05
    assert fig.layout.margin.t == 60
    if "maxheight" in getattr(fig.layout.legend, "_valid_props", set()):
        assert fig.layout.legend.maxheight == 70
    assert fig.layout.yaxis.title.text == "Energy (MWh)"
    assert fig.layout.yaxis.domain == (0, 0.95)
    assert fig.layout.xaxis.title.text == "Timestamp"


def test_summary_plot_ignores_columns_without_summary_schema_entries() -> None:
    """Summary data should only include configured summary columns."""
    df = pd.DataFrame(
        {
            "grid_consumption": [400.0, 100.0],
            "grid_feed_in": [100.0, 0.0],
            "battery_charge": [200.0, 300.0],
            "battery_discharge": [0.0, 200.0],
            "pv_asset_production": [500.0, 100.0],
            "day_ahead_price": [70.0, 80.0],
        },
        index=pd.to_datetime(["2026-02-01 00:00:00", "2026-02-01 00:30:00"]),
    )

    fig, months = summary_plot(df, period="monthly")
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert all("_pos" not in col and "_neg" not in col for col in months.columns)
    assert isclose(months.loc[months.index[0], "Netzbezug"], 0.25)
    assert isclose(months.loc[months.index[0], "Netz Einspeisung"], -0.05)
    assert isclose(months.loc[months.index[0], "Batterie Beladung"], 0.25)
    assert isclose(months.loc[months.index[0], "Batterie Entladung"], -0.1)
    assert "Netzbezug" in traces_by_name
    assert "Batterie Beladung" in traces_by_name
    assert "Netz Einspeisung" in traces_by_name
    assert "Batterie Entladung" in traces_by_name
    assert "PV-Erzeugung" in traces_by_name
    assert "Day Ahead Preis" not in traces_by_name
    assert list(traces_by_name["Netzbezug"].y) == [0.25]
    assert list(traces_by_name["Netz Einspeisung"].y) == [-0.05]
    assert "PV Verbrauch" not in traces_by_name
    assert list(traces_by_name["PV-Erzeugung"].y) == [-0.3]


def test_summary_plot_accepts_pv_as_pv_production_alias() -> None:
    """A plain pv column should be plotted as PV production in summaries."""
    df = pd.DataFrame(
        {
            "grid_consumption": [400.0, 100.0],
            "mid_consumption": [600.0, 100.0],
            "grid_feed_in": [100.0, 0.0],
            "pv": [500.0, 100.0],
            "battery_charge": [200.0, 300.0],
            "battery_discharge": [0.0, 200.0],
        },
        index=pd.to_datetime(["2026-02-01 00:00:00", "2026-02-01 00:30:00"]),
    )

    fig, summary = summary_plot(df, period="monthly")
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert isclose(summary.loc[summary.index[0], "PV-Erzeugung"], -0.3)
    assert "PV-Erzeugung" in traces_by_name
    assert list(traces_by_name["PV-Erzeugung"].y) == [-0.3]
    assert traces_by_name["PV-Erzeugung"].offsetgroup == "production"
    assert traces_by_name["Batterie Beladung"].offsetgroup == "battery"
    assert traces_by_name["Batterie Entladung"].offsetgroup == "battery"


def test_summary_plot_maps_reporting_columns_to_german_names_and_colors() -> None:
    """Reporting metric columns should not leak raw names into monthly legends."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-01 00:00:00", "2026-03-01 01:00:00"]),
            "grid_consumption": [400.0, 200.0],
            "mid_consumption": [500.0, 600.0],
            "grid_feed_in": [150.0, 50.0],
            "pv_asset_production": [300.0, 400.0],
            "chp_asset_production": [100.0, 100.0],
            "battery_charge": [100.0, 200.0],
            "battery_discharge": [75.0, 25.0],
        }
    )

    fig, months = summary_plot(df, period="monthly")
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert set(traces_by_name) == {
        "Netzbezug",
        "Batterie Beladung",
        "MID Gesamtverbrauch",
        "Netz Einspeisung",
        "Batterie Entladung",
        "PV-Erzeugung",
        "BHKW-Erzeugung",
    }
    assert all("_" not in col for col in months.columns)
    assert "MID Gesamtverbrauch" in months.columns
    assert list(traces_by_name["Netzbezug"].y) == [0.6]
    assert list(traces_by_name["Netz Einspeisung"].y) == [-0.2]
    assert list(traces_by_name["Batterie Entladung"].y) == [-0.1]
    assert isclose(traces_by_name["PV-Erzeugung"].y[0], -0.7)
    assert isclose(traces_by_name["BHKW-Erzeugung"].y[0], -0.2)
    assert traces_by_name["Netzbezug"].marker.color == COLOR_DICT["Netzbezug"]
    assert traces_by_name["Netz Einspeisung"].marker.color == COLOR_DICT["Netzbezug"]
    assert (
        traces_by_name["Batterie Beladung"].marker.color
        == COLOR_DICT["Batterie Entladung"]
    )
    assert (
        traces_by_name["Batterie Entladung"].marker.color
        == COLOR_DICT["Batterie Entladung"]
    )
    assert traces_by_name["MID Gesamtverbrauch"].marker.color == "rgba(121, 85, 72, 1)"
    assert traces_by_name["PV-Erzeugung"].marker.color == COLOR_DICT["PV-Erzeugung"]
    assert traces_by_name["BHKW-Erzeugung"].marker.color == COLOR_DICT["BHKW-Erzeugung"]


def test_summary_plot_supports_daily_weekly_monthly_and_yearly_periods() -> None:
    """Summary plot should aggregate by the selected period."""
    df = pd.DataFrame(
        {"grid_consumption": [1.0] * 400},
        index=pd.date_range("2026-01-01", periods=400, freq="D"),
    )

    expected_counts: dict[SummaryPeriod, int] = {
        "daily": 400,
        "weekly": 58,
        "monthly": 14,
        "yearly": 2,
    }
    for period, expected_count in expected_counts.items():
        fig, summary = summary_plot(df, period=period)

        assert len(summary) == expected_count
        assert fig.layout.title.text == f"{period.capitalize()} Energy"


def test_summary_plot_requires_timestamps() -> None:
    """Summary aggregation needs timestamps in the index or timestamp column."""
    df = pd.DataFrame({"grid_consumption": [1.0, 2.0]})

    try:
        summary_plot(df, period="monthly")
    except TypeError as exc:
        assert "timestamp" in str(exc)
    else:
        raise AssertionError("summary_plot should require timestamps")


def test_plot_time_series_battery_soc_uses_secondary_axis_for_soc() -> None:
    """Battery charge/discharge should share y, while SOC uses y2."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "battery_power_flow": [5.0, -4.0],
            "soc": [42.0, 53.0],
            "ignored": [10.0, 11.0],
        }
    )

    fig = plot_time_series_battery_soc(df, time_col="timestamp")
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert list(traces_by_name) == [
        "Battery Charging",
        "Battery Discharging",
        "Battery SOC (%)",
    ]
    assert traces_by_name["Battery Charging"].yaxis == "y"
    assert traces_by_name["Battery Discharging"].yaxis == "y"
    assert traces_by_name["Battery SOC (%)"].yaxis == "y2"
    assert traces_by_name["Battery Charging"].fill == "tozeroy"
    assert traces_by_name["Battery Discharging"].fill == "tozeroy"
    assert traces_by_name["Battery SOC (%)"].fill == "none"
    assert list(traces_by_name["Battery Charging"].y) == [5.0, 0.0]
    assert list(traces_by_name["Battery Discharging"].y) == [0.0, -4.0]
    assert fig.layout.yaxis.title.text == "kW"
    assert fig.layout.yaxis2.title.text == "SOC [%]"
    assert fig.layout.xaxis.rangeslider.visible is True
    assert not fig.layout.updatemenus


def test_plot_time_series_battery_soc_plots_bounds_on_secondary_axis() -> None:
    """Battery SOC bounds should be dotted traces on the SOC axis."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "battery_power_flow": [5.0, -4.0],
            "soc": [42.0, 53.0],
            "battery_soc_lower_bound_pct": [20.0, 25.0],
            "battery_soc_upper_bound_pct": [80.0, 75.0],
        }
    )

    fig = plot_time_series_battery_soc(df, time_col="timestamp")
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert list(traces_by_name) == [
        "Battery Charging",
        "Battery Discharging",
        "Battery SOC (%)",
        "Battery SOC Lower Bound (%)",
        "Battery SOC Upper Bound (%)",
    ]
    assert traces_by_name["Battery SOC Lower Bound (%)"].yaxis == "y2"
    assert traces_by_name["Battery SOC Upper Bound (%)"].yaxis == "y2"
    assert traces_by_name["Battery SOC Lower Bound (%)"].line.dash == "dot"
    assert traces_by_name["Battery SOC Upper Bound (%)"].line.dash == "dot"
    assert list(traces_by_name["Battery SOC Lower Bound (%)"].y) == [20.0, 25.0]
    assert list(traces_by_name["Battery SOC Upper Bound (%)"].y) == [80.0, 75.0]


def test_plot_time_series_battery_soc_accepts_custom_power_flow_column() -> None:
    """A caller-specific battery flow column should be split for plotting."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "battery_kw": [7.0, -2.0],
            "soc_pct": [42.0, 53.0],
        }
    )

    fig = plot_time_series_battery_soc(
        df,
        time_col="timestamp",
        battery_power_flow="battery_kw",
        soc_pct="soc_pct",
    )
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert list(traces_by_name["Battery Charging"].y) == [7.0, 0.0]
    assert list(traces_by_name["Battery Discharging"].y) == [0.0, -2.0]


def test_plot_time_series_battery_soc_and_usecase_adds_view_button() -> None:
    """A full battery dataframe should get a SOC/usecase view switch."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv": [12.0, 10.0],
            "day_ahead_price": [80.0, 95.0],
            "battery_soc_pct": [42.0, 53.0],
        }
    )

    fig = plot_time_series_battery_soc_and_usecase(
        df,
        time_col="timestamp",
        soc_pct="battery_soc_pct",
        secondary_y_cols=["day_ahead_price"],
        secondary_y_title="EUR/MWh",
        title="Lastgang Übersicht",
        dotted_cols=[
            "grid_consumption_without_battery",
            "peak_before_optimization",
            "day_ahead_price",
        ],
        stack_mode="energy_balance",
    )

    assert len(fig.layout.updatemenus) == 1
    assert fig.layout.width == 950
    assert fig.layout.margin.r == 180
    assert fig.layout.updatemenus[0].x == 1.1
    assert fig.layout.updatemenus[0].xanchor == "left"
    assert fig.layout.updatemenus[0].direction == "down"
    buttons = fig.layout.updatemenus[0].buttons
    assert [button.label for button in buttons] == [
        "Gesamt<br>Energieprofil",
        "Batterie<br>Ladezustand",
    ]

    assert buttons[0].args[1]["yaxis2"]["title"]["text"] == "EUR/MWh"
    assert buttons[1].args[1]["yaxis2"]["title"]["text"] == "SOC [%]"
    usecase_visible = list(buttons[0].args[0]["visible"])
    soc_visible = list(buttons[1].args[0]["visible"])
    assert soc_visible[:3] == [True, True, True]
    assert not any(soc_visible[3:])
    assert usecase_visible[:3] == [False, False, False]
    assert any(usecase_visible[3:])
    assert all(trace.visible is False for trace in fig.data[:3])
    assert all(trace.visible is True for trace in fig.data[3:])
    assert fig.layout.yaxis2.title.text == "EUR/MWh"
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }
    assert traces_by_name["Day Ahead Preis"].yaxis == "y2"
    assert traces_by_name["Day Ahead Preis"].line.dash == "dot"
    assert "battery_soc_pct" not in traces_by_name
    assert "Batterie SOC %" not in traces_by_name


def test_plot_time_series_battery_soc_and_usecase_uses_resampler() -> None:
    """The combined SOC/usecase plot should dynamically resample large traces."""
    rows = 100
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-09 06:30:00", periods=rows, freq="s"),
            "mid_consumption": range(rows),
            "grid_consumption": range(rows),
            "battery_power_flow": range(rows),
            "pv": range(rows),
            "day_ahead_price": range(rows),
            "soc": range(rows),
        }
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The 'generic' unit for NumPy timedelta is deprecated",
            category=DeprecationWarning,
            module="plotly_resampler.figure_resampler.utils",
        )
        fig = plot_time_series_battery_soc_and_usecase(
            df,
            time_col="timestamp",
            secondary_y_cols=["day_ahead_price"],
            secondary_y_title="EUR/MWh",
            resampler_default_n_shown_samples=10,
        )

    assert fig.__class__.__name__ == "FigureResampler"
    assert len(fig.data[0].x) == 10
    assert len(fig.hf_data) == len(fig.data)
    assert all("~" not in str(trace.name) for trace in fig.data)


def test_plot_time_series_battery_soc_and_usecase_without_secondary_usecase_axis() -> (
    None
):
    """The view switch should work when only the SOC view uses a secondary axis."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv": [12.0, 10.0],
            "soc": [42.0, 53.0],
        }
    )

    fig = plot_time_series_battery_soc_and_usecase(
        df,
        time_col="timestamp",
        enable_resampler=False,
    )

    buttons = fig.layout.updatemenus[0].buttons
    assert buttons[0].args[1]["yaxis2"]["visible"] is False
    assert buttons[1].args[1]["yaxis2"]["title"]["text"] == "SOC [%]"


def test_plot_time_series_battery_soc_and_usecase_skips_missing_day_ahead_axis() -> (
    None
):
    """A missing day-ahead price column should not block the usecase plot."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv": [12.0, 10.0],
            "soc": [42.0, 53.0],
        }
    )

    fig = plot_time_series_battery_soc_and_usecase(
        df,
        time_col="timestamp",
        secondary_y_cols=["day_ahead_price"],
        secondary_y_title="EUR/MWh",
        enable_resampler=False,
    )

    trace_names = [trace.name for trace in fig.data if getattr(trace, "name", None)]
    assert "Day Ahead Preis" not in trace_names
    assert fig.layout.updatemenus[0].buttons[0].args[1]["yaxis2"]["visible"] is False


def test_plot_time_series_battery_soc_and_usecase_without_battery_columns() -> None:
    """A non-battery dataframe should still produce the usecase plot."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "pv": [12.0, 10.0],
            "day_ahead_price": [80.0, 95.0],
        }
    )

    fig = plot_time_series_battery_soc_and_usecase(
        df,
        time_col="timestamp",
        secondary_y_cols=["day_ahead_price"],
        secondary_y_title="EUR/MWh",
        dotted_cols=["day_ahead_price"],
    )

    assert not fig.layout.updatemenus
    assert fig.layout.yaxis2.title.text == "EUR/MWh"
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }
    assert "Battery SOC (%)" not in traces_by_name
    assert "Day Ahead Preis" in traces_by_name
    assert traces_by_name["Day Ahead Preis"].yaxis == "y2"


def test_plot_time_series_battery_soc_does_not_add_view_button() -> None:
    """The standalone SOC plot should not include the usecase toggle."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "soc": [42.0, 53.0],
        }
    )

    fig = plot_time_series_battery_soc(df, time_col="timestamp")

    assert not fig.layout.updatemenus


def test_plot_time_series_battery_usecase_adds_peak_lines() -> None:
    """Peak reference lines should be plotted even when omitted from cols."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
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
            "consumption",
            "grid_consumption",
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
            "MID Gesamtverbrauch": [35.0, 21.0],
            "Netzbezug": [30.0, 25.0],
            "Batterie Leistungsfluss": [5.0, -4.0],
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
    assert "MID Gesamtverbrauch" in trace_names
    assert "Lastspitze vor optimierung" in trace_names
    assert "Lastspitze nach optimierung" in trace_names


def test_plot_time_series_battery_usecase_colors_day_ahead_price() -> None:
    """Day-ahead price should use the adapter default after display-name mapping."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [0.0, -4.0],
            "pv": [12.0, 10.0],
            "day_ahead_price": [80.0, 95.0],
        }
    )

    fig = plot_time_series_battery_usecase(
        df,
        time_col="timestamp",
        cols=["grid_consumption", "mid_consumption", "pv", "day_ahead_price"],
        secondary_y_cols=["day_ahead_price"],
    )
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert traces_by_name["Day Ahead Preis"].line.color == COLOR_DICT["day_ahead_price"]


def test_plot_time_series_battery_usecase_aligns_secondary_zero() -> None:
    """Day-ahead price zero should align with primary zero."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                    "2026-01-09 07:30:00",
                ]
            ),
            "mid_consumption": [35.0, 21.0, 30.0],
            "grid_consumption": [-50.0, 25.0, 150.0],
            "battery_power_flow": [0.0, 0.0, 0.0],
            "battery_discharge": [0.0, 0.0, 0.0],
            "battery_charge": [0.0, 0.0, 0.0],
            "day_ahead_price": [80.0, 95.0, 90.0],
        }
    )

    fig = plot_time_series_battery_usecase(
        df,
        time_col="timestamp",
        cols=["grid_consumption", "mid_consumption", "day_ahead_price"],
        secondary_y_cols=["day_ahead_price"],
    )

    primary_min, primary_max = fig.layout.yaxis.range
    secondary_min, secondary_max = fig.layout.yaxis2.range
    primary_zero_position = -primary_min / (primary_max - primary_min)
    secondary_zero_position = -secondary_min / (secondary_max - secondary_min)

    assert secondary_min < 0
    assert secondary_max > 0
    assert isclose(primary_zero_position, secondary_zero_position)


def test_plot_time_series_battery_usecase_colors_negative_grid_feed_in() -> None:
    """Negative Netzbezug values should be overlaid as Netz Einspeisung."""
    feed_in_color = "rgba(46,125,50,1)"
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                    "2026-01-09 07:30:00",
                ]
            ),
            "mid_consumption": [10.0, 8.0, 11.0],
            "grid_consumption": [4.0, -3.0, -2.0],
            "battery_power_flow": [0.0, 0.0, 0.0],
            "battery_discharge": [0.0, 0.0, 0.0],
            "battery_charge": [0.0, 0.0, 0.0],
        }
    )

    fig = plot_time_series_battery_usecase(
        df,
        time_col="timestamp",
        cols=["grid_consumption", "mid_consumption"],
        color_dict={"Netz Einspeisung": feed_in_color},
    )
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert traces_by_name["Netz Einspeisung"].line.color == feed_in_color
    feed_in_y = list(traces_by_name["Netz Einspeisung"].y)
    assert pd.isna(feed_in_y[0])
    assert feed_in_y[1:] == [-3.0, -2.0]


def test_plot_time_series_battery_usecase_adds_chp_and_wind_to_overlay_stacks() -> None:
    """CHP and wind should join a production stack anchored on MID consumption."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [0.0, -4.0],
            "pv": [12.0, 10.0],
            "chp": [4.0, 3.0],
            "wind": [7.0, 5.0],
        }
    )

    fig = plot_time_series_battery_usecase(
        df,
        time_col="timestamp",
        cols=[
            "mid_consumption",
            "grid_consumption",
            "battery_discharge",
            "battery_charge",
            "pv",
        ],
    )
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert "PV" in traces_by_name
    assert "CHP" in traces_by_name
    assert "Wind" in traces_by_name
    assert traces_by_name["CHP"].stackgroup == "production"
    assert traces_by_name["Wind"].stackgroup == "production"
    assert traces_by_name["PV"].stackgroup == "production"
    assert min(traces_by_name["PV"].y) == -12.0
    assert min(traces_by_name["CHP"].y) == -4.0
    assert min(traces_by_name["Wind"].y) == -7.0
    assert traces_by_name["Netzbezug"].stackgroup is None
    trace_names = [trace.name for trace in fig.data if getattr(trace, "name", None)]
    assert trace_names.index("Wind") < trace_names.index("CHP")
    assert trace_names.index("CHP") < trace_names.index("PV")
    assert trace_names.index("PV") < trace_names.index("Batterie Entladung")


def test_plot_time_series_battery_usecase_matches_viz_plotly_battery_stacks() -> None:
    """Battery discharge and charge should both render downward from their anchors."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [10.0, 21.0],
            "grid_consumption": [-12.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [-8.0, -4.0],
            "pv": [17.0, 0.0],
        }
    )

    fig = plot_time_series_battery_usecase(
        df,
        time_col="timestamp",
        cols=[
            "consumption",
            "grid_consumption",
            "battery_discharge",
            "battery_charge",
            "pv",
        ],
    )
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }
    battery_charge_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "name", None) == "Batterie Beladung"
    ]
    battery_charge_fill_trace = next(
        trace
        for trace in battery_charge_traces
        if getattr(trace, "fill", None) == "toself"
    )
    battery_charge_hover_trace = next(
        trace for trace in battery_charge_traces if getattr(trace, "opacity", None) == 0
    )

    assert traces_by_name["PV"].stackgroup == "production"
    assert traces_by_name["Batterie Entladung"].stackgroup == "production"
    assert traces_by_name["Batterie Entladung"].opacity == 0.3
    assert min(traces_by_name["Batterie Entladung"].y) == -5.0
    assert min(traces_by_name["PV"].y) == -17.0
    assert battery_charge_fill_trace.fill == "toself"
    assert min(battery_charge_fill_trace.y) == -20.0
    assert max(battery_charge_hover_trace.y) == 25.0
    assert min(battery_charge_hover_trace.y) == -12.0


def test_plot_time_series_battery_usecase_discharge_uses_grid_baseline_without_pv() -> (
    None
):
    """Battery discharge should still stack down from consumption without PV."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 21.0],
            "battery_power_flow": [5.0, 0.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [0.0, 0.0],
        }
    )

    fig = plot_time_series_battery_usecase(
        df,
        time_col="timestamp",
        cols=[
            "consumption",
            "grid_consumption",
            "battery_discharge",
            "battery_charge",
        ],
    )
    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }

    assert traces_by_name["Batterie Entladung"].stackgroup == "production"
    assert min(traces_by_name["Batterie Entladung"].y) == -5.0


def test_plot_time_series_battery_usecase_legacy_mode_restores_old_stacking() -> None:
    """Energy-balance mode should stack charging above all active supply traces."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [10.0, 21.0],
            "grid_consumption": [-12.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [-8.0, -4.0],
            "pv": [17.0, 0.0],
            "chp": [4.0, 3.0],
            "wind": [7.0, 5.0],
        }
    )

    fig = plot_time_series_battery_usecase(
        df,
        time_col="timestamp",
        cols=[
            "consumption",
            "grid_consumption",
            "battery_discharge",
            "battery_charge",
            "pv",
        ],
        stack_mode="energy_balance",
    )

    traces_by_name = {
        trace.name: trace for trace in fig.data if getattr(trace, "name", None)
    }
    battery_charge_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "name", None) == "Batterie Beladung"
    ]
    battery_charge_fill_trace = next(
        trace
        for trace in battery_charge_traces
        if getattr(trace, "fill", None) == "toself"
    )
    production_baseline_trace = next(
        trace
        for trace in fig.data
        if getattr(trace, "stackgroup", None) == "production"
        and getattr(trace, "name", None) is None
        and getattr(trace, "fill", None) is None
    )

    assert max(traces_by_name["Batterie Entladung"].y) == 5.0
    assert max(traces_by_name["PV"].y) == 17.0
    assert max(battery_charge_fill_trace.y) == 18.0
    assert list(production_baseline_trace.y) == [-12.0, 25.0]
    trace_names = [trace.name for trace in fig.data if getattr(trace, "name", None)]
    assert trace_names.index("Batterie Entladung") < trace_names.index("CHP")
    assert trace_names.index("CHP") < trace_names.index("Wind")
    assert trace_names.index("Wind") < trace_names.index("PV")


def test_plot_time_series_battery_usecase_energy_balance_alias() -> None:
    """Energy-balance mode should anchor charging above the full supply stack."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [10.0, 21.0],
            "grid_consumption": [-12.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "battery_discharge": [5.0, 0.0],
            "battery_charge": [-8.0, -4.0],
            "pv": [17.0, 0.0],
            "chp": [4.0, 0.0],
            "wind": [7.0, 0.0],
        }
    )

    fig = plot_time_series_battery_usecase(
        df,
        time_col="timestamp",
        cols=[
            "consumption",
            "grid_consumption",
            "battery_discharge",
            "battery_charge",
            "pv",
        ],
        stack_mode="energy_balance",
    )

    battery_charge_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "name", None) == "Batterie Beladung"
    ]
    battery_charge_fill_trace = next(
        trace
        for trace in battery_charge_traces
        if getattr(trace, "fill", None) == "toself"
    )
    battery_charge_hover_trace = next(
        trace for trace in battery_charge_traces if getattr(trace, "opacity", None) == 0
    )

    assert max(battery_charge_fill_trace.y) == 18.0
    assert min(battery_charge_hover_trace.y) == 10.0
    assert max(battery_charge_hover_trace.y) == 21.0
