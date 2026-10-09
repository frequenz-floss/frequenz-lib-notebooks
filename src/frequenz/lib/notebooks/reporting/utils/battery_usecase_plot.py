# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Helpers for battery-usecase plotting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colors as mcolors

from frequenz.lib.notebooks.reporting.utils.colors import COLOR_DICT

_DISPLAY_LABELS: dict[str, str] = {
    "grid_consumption": "Netzbezug",
    "grid_consumption_without_battery": "Netzbezug ohne Batterie",
    "day_ahead_price": "Day Ahead Preis",
    "consumption": "MID Gesamtverbrauch",
    "battery_power_flow": "Batterie Leistungsfluss",
    "battery_discharge": "Batterie Entladung",
    "battery_charge": "Batterie Beladung",
    "peak_before_optimization": "Lastspitze vor optimierung",
    "peak_after_optimization": "Lastspitze nach optimierung",
    "pv": "PV",
    "chp": "CHP",
    "wind": "Wind",
}

_PEAK_COLUMNS = ["peak_before_optimization", "peak_after_optimization"]
_EXCLUDED_USECASE_COLUMNS = {
    "battery_soc_pct",
    "battery_soc_lower_bound_pct",
    "battery_soc_upper_bound_pct",
    "soc",
}

_REQUIRED_PRODUCTION_OVERLAY_COLUMNS = ["grid_consumption"]
_REQUIRED_BATTERY_OVERLAY_COLUMNS = [
    "consumption",
    "battery_discharge",
    "battery_charge",
]

_BATTERY_OVERLAY_ALPHA = 0.3
_PRODUCTION_OVERLAY_ALPHA = 1
_MIN_BATTERY_OVERLAY_POWER = 0.25
_PRODUCTION_COLUMN_ORDER = [
    "BHKW-Erzeugung",
    "BHKW Erzeugung",
    "CHP-Production",
    "CHP Production",
    "CHP",
    "chp",
    "Wind-Erzeugung",
    "Wind Erzeugung",
    "Wind-Production",
    "Wind Production",
    "Wind",
    "wind",
    _DISPLAY_LABELS["pv"],
    "PV-Erzeugung",
    "PV Erzeugung",
    "PV-Production",
    "PV Production",
    "pv",
]

_PRODUCTION_DEFAULT_COLORS: dict[str, str] = {
    _DISPLAY_LABELS["pv"]: COLOR_DICT["PV"],
    _DISPLAY_LABELS["chp"]: COLOR_DICT["BHKW-Erzeugung"],
    _DISPLAY_LABELS["wind"]: COLOR_DICT["Wind-Erzeugung"],
}


def _resolve_battery_usecase_color(color_dict: dict[str, str], name: str) -> str:
    """Resolve display colors using battery-usecase semantics."""
    if name == _DISPLAY_LABELS["battery_discharge"]:
        return (
            color_dict.get(name)
            or color_dict.get("Battery Charge")
            or COLOR_DICT["Battery Charge"]
        )
    if name == _DISPLAY_LABELS["battery_charge"]:
        return (
            color_dict.get(name)
            or color_dict.get("Battery Discharge")
            or COLOR_DICT["Battery Discharge"]
        )
    return color_dict.get(name) or COLOR_DICT.get(name) or COLOR_DICT["PV"]


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
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
    consumption_col: str,
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
    normalize_map: dict[str, str] = {v: k for k, v in _DISPLAY_LABELS.items()}
    normalize_map.update(
        {
            battery_power_flow: "battery_power_flow",
            battery_charging: "battery_discharge",
            battery_discharging: "battery_charge",
            pv_col: "pv",
            consumption_col: "consumption",
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

    def _exclude_usecase_columns(seq: list[str] | None) -> list[str] | None:
        if seq is None:
            return None
        return [c for c in seq if c not in _EXCLUDED_USECASE_COLUMNS]

    df = df.rename(columns=normalize_map)
    cols = _rename(cols, normalize_map)
    fill_cols = _rename(fill_cols, normalize_map)
    dotted_cols = _rename(dotted_cols, normalize_map)
    plot_order = _rename(plot_order, normalize_map)
    secondary_y_cols = _rename(
        list(secondary_y_cols) if secondary_y_cols is not None else None, normalize_map
    )

    df = df.drop(columns=list(_EXCLUDED_USECASE_COLUMNS & set(df.columns)))
    cols = _exclude_usecase_columns(cols)
    fill_cols = _exclude_usecase_columns(fill_cols)
    dotted_cols = _exclude_usecase_columns(dotted_cols)
    plot_order = _exclude_usecase_columns(plot_order)
    secondary_y_cols = _exclude_usecase_columns(secondary_y_cols)

    for canonical_name in ("pv", "chp", "wind"):
        if canonical_name in df.columns:
            fill_cols = (
                [canonical_name]
                if fill_cols is None and canonical_name == "pv"
                else _ensure(fill_cols, [canonical_name])
            )
            plot_order = _ensure(plot_order, [canonical_name])

    if "consumption" in df.columns:
        # Plain reference line, not a fill — just make sure it's plotted.
        plot_order = _ensure(plot_order, ["consumption"])

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
    if secondary_y_cols is not None and _DISPLAY_LABELS["day_ahead_price"] not in df:
        secondary_y_cols = [
            col for col in secondary_y_cols if col != _DISPLAY_LABELS["day_ahead_price"]
        ]

    # Build color map with defaults
    colors = dict(color_dict or {})
    colors.setdefault(_DISPLAY_LABELS["grid_consumption"], COLOR_DICT["Netzbezug"])
    colors.setdefault(
        _DISPLAY_LABELS["battery_discharge"], COLOR_DICT["Battery Charge"]
    )
    colors.setdefault(
        _DISPLAY_LABELS["battery_charge"], COLOR_DICT["Battery Discharge"]
    )
    colors.setdefault(
        _DISPLAY_LABELS["day_ahead_price"],
        COLOR_DICT["day_ahead_price"],
    )
    for peak_col in peak_columns:
        colors.setdefault(_DISPLAY_LABELS[peak_col], COLOR_DICT["peak"])

    for display_name, default_color in _PRODUCTION_DEFAULT_COLORS.items():
        if display_name in df.columns:
            if cols is None:
                cols = [
                    c
                    for c in df.select_dtypes(include="number").columns
                    if c != time_col
                ]
            elif display_name not in cols:
                cols = [*cols, display_name]
            colors.setdefault(display_name, default_color)

    display_consumption = _DISPLAY_LABELS["consumption"]
    if display_consumption in df.columns:
        if cols is None:
            cols = [
                c for c in df.select_dtypes(include="number").columns if c != time_col
            ]
        elif display_consumption not in cols:
            cols = [*cols, display_consumption]
        colors.setdefault(
            display_consumption,
            COLOR_DICT.get(display_consumption)
            or COLOR_DICT.get("Consumption")
            or "#6c757d",
        )

    return df, cols, fill_cols, dotted_cols, plot_order, secondary_y_cols, colors


_PRODUCTION_STACK_GROUP = "production"
BatteryUsecaseStackMode = Literal["psc", "energy_balance"]


def add_battery_usecase_overlay_traces(
    fig: go.Figure,
    source_df: pd.DataFrame,
    *,
    color_dict: dict[str, str],
    yaxis_title: str,
    stack_mode: BatteryUsecaseStackMode,
) -> None:
    """Hide base traces and re-add them using viz_plotly-style stackgroups."""
    production_columns = _get_production_columns(source_df)

    hidden_names = {
        _DISPLAY_LABELS["battery_power_flow"],
        _DISPLAY_LABELS["battery_discharge"],
        _DISPLAY_LABELS["battery_charge"],
        "Battery Charge",
        "Battery Discharge",
        *production_columns,
    }
    fig.data = tuple(
        trace
        for trace in fig.data
        if not (isinstance(trace, go.Scatter) and trace.name in hidden_names)
    )

    display_grid = _DISPLAY_LABELS["grid_consumption"]
    display_consumption = _DISPLAY_LABELS["consumption"]
    entladung_name = _DISPLAY_LABELS["battery_discharge"]  # "Batterie Entladung"
    beladung_name = _DISPLAY_LABELS["battery_charge"]  # "Batterie Beladung"

    required_production = [
        _DISPLAY_LABELS[c] for c in _REQUIRED_PRODUCTION_OVERLAY_COLUMNS
    ]
    grid_with_battery: pd.Series | None = None
    consumption: pd.Series | None = None
    if all(c in source_df.columns for c in required_production):
        grid_with_battery = pd.to_numeric(source_df[display_grid], errors="coerce")
    if display_consumption in source_df.columns:
        consumption = pd.to_numeric(source_df[display_consumption], errors="coerce")

    required_battery = [_DISPLAY_LABELS[c] for c in _REQUIRED_BATTERY_OVERLAY_COLUMNS]
    if grid_with_battery is not None and consumption is not None:
        if stack_mode == "energy_balance":
            supply_columns = [entladung_name, *production_columns]
            supply_baseline = grid_with_battery
            finite_mask = grid_with_battery.notna()
            supply_direction: Literal["up", "down"] = "up"
        else:
            preferred_supply_order = [
                _DISPLAY_LABELS["wind"],
                _DISPLAY_LABELS["chp"],
                _DISPLAY_LABELS["pv"],
                entladung_name,
            ]
            supply_columns = [
                column
                for column in preferred_supply_order
                if column == entladung_name or column in production_columns
            ]
            supply_baseline = consumption
            finite_mask = consumption.notna()
            supply_direction = "down"
        _add_cumulative_supply_stack_traces(
            fig,
            source_df,
            baseline=supply_baseline,
            finite_mask=finite_mask,
            columns=supply_columns,
            color_dict=color_dict,
            yaxis_title=yaxis_title,
            direction=supply_direction,
            opacity_overrides={entladung_name: _BATTERY_OVERLAY_ALPHA},
            min_power_overrides={entladung_name: _MIN_BATTERY_OVERLAY_POWER},
        )

    if (
        all(c in source_df.columns for c in required_battery)
        and consumption is not None
    ):
        _add_battery_charge_overlay(
            fig,
            source_df,
            column=beladung_name,
            grid=grid_with_battery,
            consumption=consumption,
            color_dict=color_dict,
            yaxis_title=yaxis_title,
            stack_mode=stack_mode,
        )

    fig.update_layout(legend={"groupclick": "togglegroup"})
    _move_grid_traces_to_top(fig)


def _get_production_columns(source_df: pd.DataFrame) -> list[str]:
    """Return supported production columns in deterministic display order."""
    production_columns: list[str] = []
    for column in _PRODUCTION_COLUMN_ORDER:
        if column in source_df.columns and column not in production_columns:
            production_columns.append(column)
    return production_columns


def _add_stack_baseline(
    fig: go.Figure,
    source_df: pd.DataFrame,
    *,
    baseline: pd.Series,
    stackgroup: str,
) -> pd.Series:
    """Add an invisible baseline for a stackgroup and return its finite mask."""
    finite_baseline = baseline.notna()
    if not finite_baseline.any():
        return finite_baseline

    fig.add_trace(
        go.Scatter(
            x=source_df.index,
            y=baseline.where(finite_baseline),
            mode="lines",
            line={"color": "rgba(0,0,0,0)", "width": 0, "shape": "hv"},
            fillcolor="rgba(0,0,0,0)",
            stackgroup=stackgroup,
            showlegend=False,
            hoverinfo="skip",
            connectgaps=False,
        )
    )
    return finite_baseline


def _add_cumulative_supply_stack_traces(
    fig: go.Figure,
    source_df: pd.DataFrame,
    *,
    baseline: pd.Series,
    finite_mask: pd.Series,
    columns: list[str],
    color_dict: dict[str, str],
    yaxis_title: str,
    direction: Literal["up", "down"],
    opacity_overrides: dict[str, float] | None = None,
    min_power_overrides: dict[str, float] | None = None,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Add one cumulative supply stack anchored at the given baseline."""
    if not columns:
        return {}

    finite_mask = finite_mask & baseline.notna()
    if not finite_mask.any():
        return {}

    finite_baseline = _add_stack_baseline(
        fig,
        source_df,
        baseline=baseline.where(finite_mask),
        stackgroup=_PRODUCTION_STACK_GROUP,
    )
    if not finite_baseline.any():
        return {}

    opacity_overrides = opacity_overrides or {}
    min_power_overrides = min_power_overrides or {}
    stack_bounds: dict[str, tuple[pd.Series, pd.Series]] = {}
    current_top = baseline.where(finite_mask, other=np.nan)

    for column in columns:
        if column not in source_df.columns:
            continue

        raw = pd.to_numeric(source_df[column], errors="coerce")
        magnitude = raw.abs().where(finite_mask, other=np.nan)
        effective_magnitude = magnitude.where(finite_mask, other=np.nan)

        min_power = min_power_overrides.get(column, 0.0)
        if direction == "down":
            effective = (-effective_magnitude).where(effective_magnitude.gt(min_power))
            next_top = (current_top - effective_magnitude.fillna(0.0)).where(
                finite_mask, other=np.nan
            )
        else:
            effective = effective_magnitude.where(effective_magnitude.gt(min_power))
            next_top = (current_top + effective_magnitude.fillna(0.0)).where(
                finite_mask, other=np.nan
            )
        stack_bounds[column] = (current_top.copy(), next_top.copy())
        current_top = next_top

        color = _resolve_battery_usecase_color(color_dict, column)
        fig.add_trace(
            go.Scatter(
                x=source_df.index,
                y=effective,
                mode="lines",
                name=column,
                line={"color": color, "width": 1, "shape": "hv"},
                stackgroup=_PRODUCTION_STACK_GROUP,
                opacity=opacity_overrides.get(column, _PRODUCTION_OVERLAY_ALPHA),
                connectgaps=False,
                customdata=np.column_stack([raw.to_numpy(dtype=float)]),
                legendgroup=column,
                hovertemplate=(
                    f"<b>{column}</b>: %{{customdata[0]}} {yaxis_title}"
                    "<extra></extra>"
                ),
            )
        )
    return stack_bounds


def _add_battery_charge_overlay(
    fig: go.Figure,
    source_df: pd.DataFrame,
    *,
    column: str,
    grid: pd.Series | None,
    consumption: pd.Series,
    color_dict: dict[str, str],
    yaxis_title: str,
    stack_mode: BatteryUsecaseStackMode,
) -> None:
    """Add one battery-charge demand overlay for the selected stack mode."""
    raw = pd.to_numeric(source_df[column], errors="coerce")
    color = _resolve_battery_usecase_color(color_dict, column)
    if stack_mode == "energy_balance":
        anchor = consumption
        finite_mask = consumption.notna()
        direction: Literal["up", "down"] = "up"
    else:
        if grid is None:
            return
        anchor = grid
        finite_mask = grid.notna()
        direction = "down"
    if not finite_mask.any():
        return

    magnitude = raw.abs()
    hover_anchor = anchor.where(finite_mask, other=np.nan)
    charge_mask = finite_mask & magnitude.gt(_MIN_BATTERY_OVERLAY_POWER).fillna(False)
    if charge_mask.any():
        charge_base = anchor.where(finite_mask, other=np.nan)
        if direction == "down":
            charge_top = (anchor - magnitude.fillna(0.0)).where(
                finite_mask, other=np.nan
            )
        else:
            charge_top = (anchor + magnitude.fillna(0.0)).where(
                finite_mask, other=np.nan
            )
        charge_traces = _hv_fill_traces(
            source_df.index,
            charge_base,
            charge_top,
            charge_mask,
            color,
            column,
            alpha=_BATTERY_OVERLAY_ALPHA,
        )
        for trace in charge_traces:
            fig.add_trace(trace)

    if charge_mask.any():
        fig.add_trace(
            go.Scatter(
                x=source_df.index,
                y=hover_anchor.where(charge_mask),
                mode="lines",
                name=column,
                line={"color": color, "width": 0, "shape": "hv"},
                opacity=0,
                showlegend=False,
                connectgaps=False,
                customdata=np.column_stack([raw.to_numpy(dtype=float)]),
                legendgroup=column,
                hovertemplate=(
                    f"<b>{column}</b>: %{{customdata[0]}} {yaxis_title}<extra></extra>"
                ),
            )
        )


def _move_grid_traces_to_top(fig: go.Figure) -> None:
    """Move reference-line traces to the end so they render above overlays."""
    top_names = {
        _DISPLAY_LABELS["grid_consumption"],
        _DISPLAY_LABELS["consumption"],
        "Netz Einspeisung",
    }
    traces = list(fig.data)
    base_traces = [
        trace for trace in traces if getattr(trace, "name", None) not in top_names
    ]
    top_traces = [
        trace for trace in traces if getattr(trace, "name", None) in top_names
    ]
    fig.data = tuple(base_traces + top_traces)


def _with_alpha(color: str | None, alpha: float) -> str | None:
    """Return color as rgba string with the given alpha, or None if invalid."""
    if not color:
        return None
    try:
        parsed = color.strip().lower()
        if parsed.startswith("rgba(") and parsed.endswith(")"):
            parts = [part.strip() for part in parsed[5:-1].split(",")]
            if len(parts) == 4:
                r, g, b = (float(parts[0]), float(parts[1]), float(parts[2]))
                return (
                    f"rgba({int(round(r))},{int(round(g))},{int(round(b))},{alpha:.3f})"
                )
        if parsed.startswith("rgb(") and parsed.endswith(")"):
            parts = [part.strip() for part in parsed[4:-1].split(",")]
            if len(parts) == 3:
                r, g, b = (float(parts[0]), float(parts[1]), float(parts[2]))
                return (
                    f"rgba({int(round(r))},{int(round(g))},{int(round(b))},{alpha:.3f})"
                )
        r, g, b, _ = mcolors.to_rgba(color)
    except ValueError:
        return None
    return (
        f"rgba({int(round(r*255))},{int(round(g*255))},{int(round(b*255))},{alpha:.3f})"
    )


# pylint: disable=too-many-positional-arguments
def _hv_fill_traces(
    x: pd.Index,
    base: pd.Series,
    top: pd.Series,
    mask: pd.Series,
    color: str,
    name: str,
    *,
    alpha: float = 1.0,
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
                fillcolor=_with_alpha(color, alpha) or color,
                line={"color": color, "width": 1, "shape": "hv"},
                mode="lines",
                hoverinfo="skip",
                name=name if not traces else None,
                showlegend=not traces,
                legendgroup=name,
            )
        )

    return traces
