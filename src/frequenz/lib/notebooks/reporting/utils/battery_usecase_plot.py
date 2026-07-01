# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Helpers for battery-usecase plotting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colors as mcolors

from frequenz.lib.notebooks.reporting.utils.colors import COLOR_DICT

_DISPLAY_LABELS: dict[str, str] = {
    "grid_consumption": "Netzbezug",
    "grid_consumption_without_battery": "Netzbezug ohne Batterie",
    "battery_power_flow": "Batterie Leistungsfluss",
    "battery_discharge": "Batterie Entladung",
    "battery_charge": "Batterie Beladung",
    "peak_before_optimization": "Lastspitze vor optimierung",
    "peak_after_optimization": "Lastspitze nach optimierung",
    "pv": "PV",
}

_PEAK_COLUMNS = ["peak_before_optimization", "peak_after_optimization"]

_REQUIRED_OVERLAY_COLUMNS = [
    "grid_consumption",
    "grid_consumption_without_battery",
    "battery_discharge",
    "battery_charge",
]


# pylint: disable=too-many-arguments, too-many-locals
def prepare_battery_usecase_plot(
    df: pd.DataFrame,
    *,
    cols: list[str] | None,
    fill_cols: list[str] | None,
    dotted_cols: list[str] | None,
    plot_order: list[str] | None,
    secondary_y_cols: Sequence[str] | None,
    color_dict: dict[str, str] | None,
    time_col: str | None,
    battery_power_flow: str,
    battery_charging: str,
    battery_discharging: str,
    pv_col: str,
    grid_consumption_without_battery: str,
    grid_consumption: str,
) -> tuple[
    pd.DataFrame,
    list[str] | None,
    list[str] | None,
    list[str] | None,
    list[str] | None,
    list[str] | None,
    dict[str, str],
]:
    """Normalize battery-usecase inputs, apply plot defaults, and build color map."""
    # Build rename map: legacy display names + custom column names → canonical
    normalize_map: dict[str, str] = {v: k for k, v in _DISPLAY_LABELS.items()}
    normalize_map.update(
        {
            battery_power_flow: "battery_power_flow",
            battery_charging: "battery_discharge",
            battery_discharging: "battery_charge",
            pv_col: "pv",
            grid_consumption_without_battery: "grid_consumption_without_battery",
            grid_consumption: "grid_consumption",
        }
    )

    def _rename(seq: list[str] | None, mapping: dict[str, str]) -> list[str] | None:
        return None if seq is None else [mapping.get(item, item) for item in seq]

    def _ensure(seq: list[str] | None, items: list[str]) -> list[str] | None:
        if seq is None:
            return None
        result = list(seq)
        for item in items:
            if item not in result:
                result.append(item)
        return result

    df = df.rename(columns=normalize_map)
    cols = _rename(cols, normalize_map)
    fill_cols = _rename(fill_cols, normalize_map)
    dotted_cols = _rename(dotted_cols, normalize_map)
    plot_order = _rename(plot_order, normalize_map)
    secondary_y_cols = _rename(
        list(secondary_y_cols) if secondary_y_cols is not None else None, normalize_map
    )

    if "pv" in df.columns:
        fill_cols = ["pv"] if fill_cols is None else _ensure(fill_cols, ["pv"])
        plot_order = _ensure(plot_order, ["pv"])

    peak_columns = [c for c in _PEAK_COLUMNS if c in df.columns]
    cols = _ensure(cols, peak_columns)
    plot_order = _ensure(plot_order, peak_columns)
    dotted_cols = _ensure(dotted_cols, peak_columns) or peak_columns

    # Localize canonical names → display labels
    df = df.rename(columns=_DISPLAY_LABELS)
    cols = _rename(cols, _DISPLAY_LABELS)
    fill_cols = _rename(fill_cols, _DISPLAY_LABELS)
    dotted_cols = _rename(dotted_cols, _DISPLAY_LABELS)
    plot_order = _rename(plot_order, _DISPLAY_LABELS)
    secondary_y_cols = _rename(secondary_y_cols, _DISPLAY_LABELS)

    # Build color map with defaults
    colors = dict(color_dict or {})
    colors.setdefault(_DISPLAY_LABELS["grid_consumption"], COLOR_DICT["Netzbezug"])
    colors.setdefault(
        _DISPLAY_LABELS["grid_consumption_without_battery"], COLOR_DICT["Netzbezug"]
    )
    for peak_col in peak_columns:
        colors.setdefault(_DISPLAY_LABELS[peak_col], COLOR_DICT["peak"])

    display_pv = _DISPLAY_LABELS["pv"]
    if display_pv in df.columns:
        if cols is None:
            cols = [
                c for c in df.select_dtypes(include="number").columns if c != time_col
            ]
        elif display_pv not in cols:
            cols = [*cols, display_pv]
        colors.setdefault(display_pv, COLOR_DICT["PV"])

    return df, cols, fill_cols, dotted_cols, plot_order, secondary_y_cols, colors


# pylint: disable=too-many-locals
def add_battery_usecase_overlay_traces(
    fig: go.Figure,
    source_df: pd.DataFrame,
    *,
    color_dict: dict[str, str],
    yaxis_title: str,
) -> None:
    """Hide base battery traces and add charge/discharge filled overlay areas."""
    # Hide base traces replaced by overlay fills
    hidden_names = {
        _DISPLAY_LABELS["battery_power_flow"],
        _DISPLAY_LABELS["battery_discharge"],
        _DISPLAY_LABELS["battery_charge"],
        "Battery Charge",
        "Battery Discharge",
    }
    for trace in fig.data:
        if isinstance(trace, go.Scatter) and trace.name in hidden_names:
            trace.showlegend = False
            trace.hoverinfo = "skip"
            trace.fill = "none"
            trace.line = {
                "color": "rgba(0,0,0,0)",
                "width": 0,
                "shape": "hv",
            }

    # Check all required columns are present before adding overlays
    required = [_DISPLAY_LABELS[c] for c in _REQUIRED_OVERLAY_COLUMNS]
    if not all(c in source_df.columns for c in required):
        return

    display_grid = _DISPLAY_LABELS["grid_consumption"]
    display_grid_no_batt = _DISPLAY_LABELS["grid_consumption_without_battery"]
    charge_name = _DISPLAY_LABELS["battery_discharge"]
    discharge_name = _DISPLAY_LABELS["battery_charge"]

    charge_color = (
        color_dict.get(charge_name)
        or color_dict.get("Battery Charge")
        or COLOR_DICT["Battery Charge"]
    )
    discharge_color = (
        color_dict.get(discharge_name)
        or color_dict.get("Battery Discharge")
        or COLOR_DICT["Battery Discharge"]
    )

    grid_with_battery = pd.to_numeric(source_df[display_grid], errors="coerce")
    grid_without_battery = pd.to_numeric(
        source_df[display_grid_no_batt], errors="coerce"
    )
    charging = pd.to_numeric(source_df[charge_name], errors="coerce")
    discharging = pd.to_numeric(source_df[discharge_name], errors="coerce")

    finite_grid = grid_with_battery.notna() & grid_without_battery.notna()
    charge_mask = finite_grid & charging.gt(0.0).fillna(False)
    discharge_mask = finite_grid & discharging.lt(0.0).fillna(False)

    for trace in _hv_fill_traces(
        source_df.index,
        grid_without_battery.where(charge_mask),
        grid_with_battery.where(charge_mask),
        charge_mask,
        charge_color,
        charge_name,
    ):
        fig.add_trace(trace)

    for trace in _hv_fill_traces(
        source_df.index,
        grid_with_battery.where(discharge_mask),
        grid_without_battery.where(discharge_mask),
        discharge_mask,
        discharge_color,
        discharge_name,
    ):
        fig.add_trace(trace)

    fig.add_trace(
        go.Scatter(
            x=source_df.index,
            y=grid_with_battery.where(charge_mask),
            mode="lines",
            line={
                "color": "rgba(0,0,0,0)",
                "width": 0,
                "shape": "hv",
            },
            showlegend=False,
            connectgaps=False,
            customdata=np.column_stack([charging.to_numpy(dtype=float)]),
            legendgroup=charge_name,
            hovertemplate=f"<b>{charge_name}</b>: %{{customdata[0]}} {yaxis_title}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=source_df.index,
            y=grid_with_battery.where(discharge_mask),
            mode="lines",
            line={
                "color": "rgba(0,0,0,0)",
                "width": 0,
                "shape": "hv",
            },
            showlegend=False,
            connectgaps=False,
            customdata=np.column_stack([discharging.to_numpy(dtype=float)]),
            legendgroup=discharge_name,
            hovertemplate=(
                f"<b>{discharge_name}</b>: "
                f"%{{customdata[0]}} {yaxis_title}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        legend={
            "groupclick": "togglegroup",
        }
    )


def _with_alpha(color: str | None, alpha: float) -> str | None:
    """Return color as rgba string with the given alpha, or None if invalid."""
    if not color:
        return None
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except ValueError:
        return None
    return (
        f"rgba({int(round(r*255))},{int(round(g*255))},{int(round(b*255))},{alpha:.3f})"
    )


# pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
def _hv_fill_traces(
    x: pd.Index,
    base: pd.Series,
    top: pd.Series,
    mask: pd.Series,
    color: str,
    name: str,
) -> list[go.Scatter]:
    """Build filled step polygons for contiguous active intervals."""
    if len(x) < 2:
        return []

    interval_mask = mask.to_numpy(dtype=bool)
    x_values = list(x)
    base_values = base.to_numpy(dtype=float)
    top_values = top.to_numpy(dtype=float)
    active_intervals = np.flatnonzero(interval_mask[:-1])
    if len(active_intervals) == 0:
        return []

    breaks = np.where(np.diff(active_intervals) > 1)[0] + 1
    groups = np.split(active_intervals, breaks)
    traces: list[go.Scatter] = []

    def _step_coords(
        left_edges: list[object], right_edge: object, values: np.ndarray
    ) -> tuple[list[object], list[float]]:
        step_x = [left_edges[0]]
        step_y = [float(values[0])]
        for idx, value in enumerate(values):
            step_x.append(right_edge if idx == len(values) - 1 else left_edges[idx + 1])
            step_y.append(float(value))
            if idx < len(values) - 1:
                step_x.append(left_edges[idx + 1])
                step_y.append(float(values[idx + 1]))
        return step_x, step_y

    for group in groups:
        left_edges = [x_values[idx] for idx in group]
        right_edge = x_values[group[-1] + 1]
        top_x, top_y = _step_coords(left_edges, right_edge, top_values[group])
        base_x, base_y = _step_coords(left_edges, right_edge, base_values[group])
        traces.append(
            go.Scatter(
                x=top_x + base_x[::-1],
                y=top_y + base_y[::-1],
                fill="toself",
                fillcolor=_with_alpha(color, 1) or color,
                line={"color": color, "width": 1.5, "shape": "hv"},
                mode="lines",
                name=name if not traces else None,
                showlegend=not traces,
                hoverinfo="skip",
                legendgroup=name,
            )
        )

    return traces
