# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Visualization for asset optimization reporting using Plotly."""

import logging

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .viz_colors import (
    AVAILABLE,
    BUY,
    CHARGE,
    CHP,
    CONSUMPTION,
    DISCHARGE,
    GRID,
    PV,
    SELL,
    SOC,
    TRANSPARENT,
    ZERO_LINE,
)
from .viz_data import (
    prepare_battery_power_data,
    prepare_energy_trade_data,
    prepare_monthly_data,
    prepare_power_flow_data,
)

_logger = logging.getLogger(__name__)


FIGURE_SIZE = (1000, 700)
LINE_WIDTH = 1
FONT_SIZE = 14
FONT_FAMILY = "Golos Text, sans-serif"
AXIS_LINE_WIDTH = 1.5
GRID_LINE_WIDTH = 1
HOVER_BG = "rgba(255,255,255,0.95)"
HOVER_BORDER = "rgba(0,0,0,0.2)"
HOVER_FONT_COLOR = "#111"
LEGEND_COLS = 3


def _legend_grid(num_items: int, *, cols: int = LEGEND_COLS) -> dict[str, int]:
    items = max(num_items, 1)
    cols_eff = min(cols, items)
    rows = (items + cols_eff - 1) // cols_eff
    return {"rows": rows, "cols": cols_eff}


def _legend_config(
    *,
    num_items: int,
    y: float,
    x: float = 0.0,
    cols: int = LEGEND_COLS,
) -> dict[str, object]:
    grid = _legend_grid(num_items, cols=cols)
    config: dict[str, object] = {
        "orientation": "h",
        "yanchor": "top",
        "y": y,
        "xanchor": "left",
        "x": x,
        "bgcolor": "rgba(255,255,255,0)",
    }
    config["entrywidth"] = 1.0 / grid["cols"]
    config["entrywidthmode"] = "fraction"
    return config


def _apply_common_layout(
    fig: go.Figure,
    *,
    y_title: str | None = None,
    legend_items: int | None = None,
) -> None:
    fig.update_layout(
        width=FIGURE_SIZE[0],
        height=FIGURE_SIZE[1],
        template="plotly_white",
        font={"size": FONT_SIZE, "family": FONT_FAMILY},
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel={
            "bgcolor": HOVER_BG,
            "bordercolor": HOVER_BORDER,
            "font": {
                "family": FONT_FAMILY,
                "size": FONT_SIZE,
                "color": HOVER_FONT_COLOR,
            },
            "align": "left",
            "namelength": -1,
        },
        separators=",.",
    )
    if legend_items is not None:
        fig.update_layout(legend=_legend_config(num_items=legend_items, y=1.05))
    if y_title:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(
        showgrid=True,
        showline=False,
        mirror=False,
        gridwidth=GRID_LINE_WIDTH,
        ticks="outside",
        ticklen=6,
        tickcolor="rgba(0,0,0,0.35)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        showline=False,
        mirror=False,
        gridwidth=GRID_LINE_WIDTH,
        ticks="outside",
        ticklen=6,
        tickcolor="rgba(0,0,0,0.35)",
        zeroline=False,
    )
    fig.update_traces(
        line={"width": LINE_WIDTH, "simplify": False}, selector={"type": "scatter"}
    )


# pylint: disable=too-many-arguments
def _apply_axis_padding(
    fig: go.Figure,
    *,
    x_index: pd.Index,
    y_min: float,
    y_max: float,
    row: int | None = None,
    col: int | None = None,
    secondary_y: bool | None = None,
) -> None:
    if len(x_index) > 1:
        span = x_index.max() - x_index.min()
        pad = span * 0.1
        fig.update_xaxes(range=[x_index.min(), x_index.max() + pad], row=row, col=col)

    padded_max = y_max * 1.1 if y_max >= 0 else y_max * 0.9
    fig.update_yaxes(
        range=[y_min, padded_max], row=row, col=col, secondary_y=secondary_y
    )


def _apply_range_slider(
    fig: go.Figure, *, row: int | None = None, col: int | None = None
) -> None:
    fig.update_xaxes(
        rangeslider={"visible": True, "thickness": 0.08},
        row=row,
        col=col,
    )


def _add_zero_boundaries(series: pd.Series) -> pd.Series:
    """
    For each transition valid→NaN or NaN→valid, insert a 0-value point
    at that boundary so fill='tozeroy' closes cleanly instead of bridging.
    """
    s = series.copy().astype(float)
    is_null = s.isna()

    # Index positions just before a gap (last valid before NaN run)
    before_gap = (~is_null) & is_null.shift(-1, fill_value=False)
    # Index positions just after a gap (first valid after NaN run)
    after_gap  = (~is_null) & is_null.shift(1, fill_value=False)

    s[before_gap] = 0.0
    s[after_gap]  = 0.0
    return s


def plot_power_flow(
    df: pd.DataFrame,
    fig: go.Figure | None = None,
    row: int | None = None,
) -> go.Figure:
    """Plot the microgrid power flow as a stacked time-series chart.

    Builds a Plotly figure showing the evolution of key power-flow components
    over time, including on-site production (CHP and/or PV), battery charging and
    discharging (when available), grid exchange, and total consumption. The input
    data is normalized via ``prepare_power_flow_data`` before traces are added.

    If no figure is provided, a new one is created. When ``row`` is not provided,
    the function also applies the common layout and a range slider. Axis padding is
    applied in all cases based on the combined y-range of the plotted series.

    Args:
        df: Input DataFrame containing the columns required by
            ``prepare_power_flow_data``.
        fig: Optional existing figure to add traces to. If not provided, a new
            ``go.Figure`` is created.
        row: Optional subplot row index. When provided, common layout and range
            slider configuration are skipped and axis padding is applied to the
            specified subplot row.

    Returns:
        A Plotly figure containing the power-flow traces.
    """
    data = prepare_power_flow_data(df)

    if fig is None:
        fig = go.Figure()

    legend_items = 0

    if data.has_chp:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.chp,
                name="CHP",
                stackgroup="production",
                line={"color": CHP, "shape": "linear"},
                opacity=0.8,
                hovertemplate="<b>CHP</b>: %{y} kW<extra></extra>",
            )
        )
        legend_items += 1

    if data.has_pv:
        pv_label = "PV (on CHP)" if data.has_chp else "PV"
        pv_series = data.production - data.chp
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=pv_series,
                name=pv_label,
                stackgroup="production",
                line={"color": PV, "shape": "linear"},
                opacity=0.9,
                hovertemplate="<b>%{fullData.name}</b>: %{y} kW<extra></extra>",
            )
        )
        legend_items += 1

    if data.charge is not None and data.discharge is not None:
        # 1. Consumption Reference (The 'Floor' for Charge, 'Ceiling' for Discharge)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.consumption,
                stackgroup="charge_stack",
                fill="none",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        charge_delta = (data.charge - data.consumption).clip(lower=0)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=charge_delta,
                name="Charge",
                stackgroup="charge_stack",
                line={"color": CHARGE, "shape": "linear", "width": 1},
                customdata=data.charge,
                hovertemplate="<b>Charge</b>: %{customdata} kW<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.consumption,
                stackgroup="discharge_stack",
                fill="none",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        discharge_delta = (data.discharge - data.consumption).clip(upper=0)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=discharge_delta,
                name="Discharge",
                stackgroup="discharge_stack",
                line={"color": DISCHARGE, "shape": "linear", "width": 1},
                customdata=data.discharge,
                hovertemplate="<b>Discharge</b>: %{customdata} kW<extra></extra>",
            )
        )
        legend_items += 2

    if data.grid is not None:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.grid,
                name="Grid",
                line={"color": GRID, "shape": "hv"},
                hovertemplate="<b>Grid</b>: %{y} kW<extra></extra>",
            )
        )
        legend_items += 1

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.consumption,
            name="Consumption",
            line={"color": CONSUMPTION, "shape": "hv"},
            hovertemplate="<b>Consumption</b>: %{y} kW<extra></extra>",
        )
    )
    legend_items += 1

    if row is None:
        _apply_common_layout(fig, y_title="Power (kW)", legend_items=legend_items)
        fig.update_layout(title={"text": "Power Flow", "x": 0.0, "xanchor": "left"})
        _apply_range_slider(fig)

    y_series = [data.consumption, data.chp, data.production]
    if data.grid is not None:
        y_series.append(data.grid)
    if data.charge is not None:
        y_series.append(data.charge)
    if data.discharge is not None:
        y_series.append(data.discharge)
    y_all = pd.concat(y_series)
    _apply_axis_padding(
        fig,
        x_index=data.index,
        y_min=float(y_all.min()),
        y_max=float(y_all.max()),
        row=row,
        col=1 if row is not None else None,
    )
    return fig


def plot_energy_trade(
    df: pd.DataFrame,
    fig: go.Figure | None = None,
    row: int | None = None,
) -> go.Figure:
    """Plot the microgrid energy trade as a time-series chart.

    Creates a Plotly figure showing bought and sold energy over time based on
    the processed output of ``prepare_energy_trade_data``. The function adds
    separate traces for energy buy and sell values and optionally applies the
    standard layout and range slider when used as a standalone figure.

    Args:
        df: Input DataFrame containing the columns required by
            ``prepare_energy_trade_data``.
        fig: Optional existing figure to which traces are added. If not provided,
            a new ``go.Figure`` is created.
        row: Optional subplot row index. When provided, the common layout and
            range slider are not applied, and axis padding is scoped to the
            specified subplot row.

    Returns:
        A Plotly figure containing the energy trade traces.
    """
    data = prepare_energy_trade_data(df)

    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.buy,
            name="Buy",
            line={"color": BUY, "shape": "hv"},
            fill="tozeroy",
            hovertemplate="<b>Buy</b>: %{y} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.sell,
            name="Sell",
            line={"color": SELL, "shape": "hv"},
            fill="tozeroy",
            hovertemplate="<b>Sell</b>: %{y} kWh<extra></extra>",
        )
    )

    if row is None:
        _apply_common_layout(fig, y_title="Energy (kWh)", legend_items=2)
        fig.update_layout(title={"text": "Energy Trade", "x": 0.0, "xanchor": "left"})
        _apply_range_slider(fig)
    y_all = pd.concat([data.buy, data.sell])
    _apply_axis_padding(
        fig,
        x_index=data.index,
        y_min=float(y_all.min()),
        y_max=float(y_all.max()),
        row=row,
        col=1 if row is not None else None,
    )
    return fig


def plot_power_flow_trade(df: pd.DataFrame) -> go.Figure:
    """Create a combined subplot showing power flow and energy trade.

    Builds a two-row Plotly figure with shared x-axis:
    - The top subplot renders the power-flow visualization.
    - The bottom subplot renders the corresponding energy-trade visualization.

    Traces from the underlying figures are merged into a single subplot layout and
    assigned to two separate legends (one per subplot). The figure applies a common
    layout configuration, axis styling, subplot-specific y-axis titles, and adds a
    range slider on the lower subplot.

    Args:
        df: Input DataFrame containing the columns required by ``plot_power_flow``
            and ``plot_energy_trade``.

    Returns:
        A Plotly figure containing the stacked power-flow and energy-trade
        subplots with independent legends and shared time navigation.
    """
    fig_final = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        row_heights=[0.75, 0.25],
    )

    # Generate standalone figures
    fig_power = plot_power_flow(df)
    fig_trade = plot_energy_trade(df)

    for trace in fig_power.data:
        trace.legend = "legend"
        fig_final.add_trace(trace, row=1, col=1)

    for trace in fig_trade.data:
        trace.legend = "legend2"
        fig_final.add_trace(trace, row=2, col=1)

    # Apply "Common Layout" Logic manually to the Subplot
    fig_final.update_layout(
        width=FIGURE_SIZE[0],
        height=FIGURE_SIZE[1],
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        font={"size": FONT_SIZE, "family": FONT_FAMILY},
        margin={"l": 30, "r": 70, "t": 30, "b": 40},
        hoverlabel={
            "bgcolor": HOVER_BG,
            "bordercolor": HOVER_BORDER,
            "font": {
                "family": FONT_FAMILY,
                "size": FONT_SIZE,
                "color": HOVER_FONT_COLOR,
            },
            "align": "left",
            "namelength": -1,
        },
        separators=",.",
    )

    # Configure the layout for both legends
    power_items = len([t for t in fig_power.data if t.showlegend is not False])
    trade_items = len([t for t in fig_trade.data if t.showlegend is not False])
    fig_final.update_layout(
        legend=_legend_config(num_items=power_items, y=1.1),
        legend2=_legend_config(num_items=trade_items, y=0.30),
    )

    fig_final.update_xaxes(
        showgrid=True,
        showline=False,
        mirror=False,
        gridwidth=GRID_LINE_WIDTH,
        ticks="outside",
        ticklen=6,
        tickcolor="rgba(0,0,0,0.35)",
        zeroline=False,
    )

    fig_final.update_yaxes(
        showgrid=True,
        showline=False,
        mirror=False,
        gridwidth=GRID_LINE_WIDTH,
        ticks="outside",
        ticklen=6,
        tickcolor="rgba(0,0,0,0.35)",
        zeroline=False,
    )
    fig_final.update_traces(line={"width": LINE_WIDTH, "simplify": False})

    # Set Specific Titles
    fig_final.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig_final.update_yaxes(title_text="Energy (kWh)", row=2, col=1)
    _apply_range_slider(fig_final, row=2, col=1)

    return fig_final


def plot_battery_power(df: pd.DataFrame) -> go.Figure:
    """Plot battery power and state of charge (SOC) over time.

    Creates a Plotly figure visualizing battery behavior, including available
    power, charging, discharging, and SOC. The SOC is displayed on a secondary
    y-axis, while power-related traces share the primary axis.

    Args:
        df: Input DataFrame containing the columns required by
            ``prepare_battery_power_data``.

    Returns:
        A Plotly figure showing battery power flows and SOC.
    """
    data = prepare_battery_power_data(df)

    fig = go.Figure()

    charge_pos = data.charge.clip(lower=0)
    available_pos = data.available.clip(lower=0)
    charge_plot = charge_pos.where(charge_pos > 0)
    charge_plot = charge_plot.where(charge_plot.notna(), None)
    mask = charge_plot.notna() & available_pos.notna()
    charge_plot = charge_plot.where(
        ~mask, charge_plot.where(charge_plot <= available_pos, available_pos)
    )
    discharge_plot = data.battery.where(data.discharge.notna(), None)

    charge_plot = _add_zero_boundaries(charge_plot)
    discharge_plot = _add_zero_boundaries(discharge_plot)

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.soc,
            name="SOC",
            line={"color": SOC, "shape": "linear"},
            fill="tozeroy",
            opacity=0.4,
            yaxis="y2",
            hovertemplate="<b>%{fullData.name}</b>: %{y}%<extra></extra>",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.available,
            name="Available power",
            line={"color": AVAILABLE, "shape": "linear"},
            hovertemplate="<b>%{fullData.name}</b>: %{y} kW<extra></extra>",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=[0] * len(data.index),
            name="Zero",
            line={"color": ZERO_LINE, "dash": "dash", "shape": "linear"},
            showlegend=False,
            hoverinfo="skip",
        ),
    )

    # Fill trace (no line, so no connecting line across gaps)
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=charge_plot,
            mode="none",
            fill="tozeroy",
            fillcolor=CHARGE,
            connectgaps=False,
            showlegend=False,
            hoverinfo="skip",
            line={"shape": "linear"},
        ),
    )

    # Line trace (carries the legend entry and tooltip)
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=charge_plot,
            name="Charge",
            line={"color": CHARGE, "shape": "linear"},
            connectgaps=False,
            hovertemplate="<b>%{fullData.name}</b>: %{y} kW<extra></extra>",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=discharge_plot,
            mode="none",
            fill="tozeroy",
            fillcolor=DISCHARGE,
            connectgaps=False,
            showlegend=False,
            hoverinfo="skip",
            line={"shape": "linear"},
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=discharge_plot,
            name="Discharge",
            line={"color": DISCHARGE, "shape": "linear"},
            connectgaps=False,
            opacity=0.9,
            hovertemplate="<b>%{fullData.name}</b>: %{y} kW<extra></extra>",
        ),
    )

    fig.update_layout(
        yaxis={
            "title": "Battery Power",
            "range": [-data.max_abs_battery * 1.1, data.max_abs_battery * 1.1],
        },
        yaxis2={
            "title": "Battery SOC",
            "range": [0, 100],
            "overlaying": "y",
            "side": "right",
            "tickfont": {"color": SOC},
        },
    )

    _apply_common_layout(fig, legend_items=4)
    fig.update_layout(legend=_legend_config(num_items=4, y=1.1, cols=4))
    _apply_range_slider(fig)
    return fig


def plot_monthly(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    """Plot monthly aggregated energy data as grouped bar charts.

    Builds a Plotly bar chart showing positive and negative monthly energy
    values returned by ``prepare_monthly_data``. Positive values are plotted
    as standard bars, while negative values are rendered with reduced opacity
    for visual distinction.

    Args:
        df:
            Input DataFrame containing time-series data required by
            ``prepare_monthly_data``.

    Returns:
        The monthly aggregated DataFrame used for plotting.
    """
    data = prepare_monthly_data(df)

    fig = go.Figure()
    x_labels = pd.to_datetime(data.months.index).strftime("%d-%m-%Y")
    for column in data.positive.columns:
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=data.positive[column],
                name=column,
                text=data.positive[column].round(3),
                textposition="outside",
                offsetgroup=column,
            )
        )
    for column in data.negative.columns:
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=data.negative[column],
                name=column,
                opacity=0.7,
                text=data.negative[column].round(3),
                textposition="outside",
                offsetgroup=column,
            )
        )

    _apply_common_layout(fig, y_title="Energy (MWh)")
    fig.update_layout(
        barmode="group",
        title={
            "text": "Monthly Energy",
            "x": 0.05,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "pad": {"t": 8},
        },
    )
    fig.update_xaxes(
        tickangle=45, showdividers=False, dividercolor="rgba(0,0,0,0)", automargin=True
    )
    return fig, data.months
