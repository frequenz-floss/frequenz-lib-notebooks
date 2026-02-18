# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Visualization for asset optimization reporting using Plotly."""

import logging

import pandas as pd
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


FIGURE_SIZE = (900, 500)


def _apply_common_layout(fig: go.Figure, *, y_title: str | None = None) -> None:
    fig.update_layout(
        width=FIGURE_SIZE[0],
        height=FIGURE_SIZE[1],
        template="plotly_white",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.05,
            "bordercolor": "black",
            "borderwidth": 1,
        },
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    if y_title:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(
        showgrid=True,
        showline=True,
        mirror=True,
        linecolor="black",
    )
    fig.update_yaxes(showgrid=True, showline=True, mirror=True, linecolor="black")


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

    if data.has_chp:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.chp,
                name="CHP",
                line={"color": CHP},
                fill="tozeroy",
                opacity=0.5,
                hovertemplate="<b>CHP</b>: %{y} kW<extra></extra>",
            )
        )

    if data.has_pv:
        pv_label = "PV (on CHP)" if data.has_chp else "PV"
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.production,
                name=pv_label,
                line={"color": PV},
                fill="tonexty" if data.has_chp else "tozeroy",
                opacity=0.7,
                hovertemplate="<b>%{fullData.name}</b>: %{y} kW<extra></extra>",
            )
        )

    if data.charge is not None and data.discharge is not None:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.consumption,
                name="Consumption (base)",
                line={"color": TRANSPARENT},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.charge,
                name="Charge",
                line={"color": CHARGE},
                opacity=0.2,
                hovertemplate="<b>Charge</b>: %{y} kW<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.consumption,
                name="Consumption (base)",
                line={"color": TRANSPARENT},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.discharge,
                name="Discharge",
                line={"color": DISCHARGE},
                opacity=0.5,
                hovertemplate="<b>Discharge</b>: %{y} kW<extra></extra>",
            )
        )

    if data.grid is not None:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.grid,
                name="Grid",
                line={"color": GRID},
                hovertemplate="<b>Grid</b>: %{y} kW<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.consumption,
            name="Consumption",
            line={"color": CONSUMPTION},
            hovertemplate="<b>Consumption</b>: %{y} kW<extra></extra>",
        )
    )

    if row is None:
        _apply_common_layout(fig, y_title="Power (kW)")
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
        _apply_common_layout(fig, y_title="Energy (kWh)")
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
    fig_final = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
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
        margin={"l": 60, "r": 150, "t": 40, "b": 40}, # Increased 'r' for legends
    )

    # Apply Multi-Legend Formatting
    legend_style = {
        "orientation": "v",
        "xanchor": "left",
        "x": 1.05,
        "bordercolor": "black",
        "borderwidth": 1,
    }

    # Configure the layout for both legends
    fig_final.update_layout(
        legend={**legend_style, "y": 1, "yanchor": "top", "title": "Power Flow"},
        legend2={**legend_style, "y": 0.05, "yanchor": "bottom", "title": "Energy Trade"},
    )

    fig_final.update_xaxes(
        showgrid=True, showline=True, mirror=True, linecolor="black"
    )

    fig_final.update_yaxes(
        showgrid=True, showline=True, mirror=True, linecolor="black"
    )

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

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.soc,
            name="SOC",
            line={"color": SOC},
            fill="tozeroy",
            opacity=0.4,
            yaxis="y2",
            hovertemplate="<b>SOC</b>: %{y}%<extra></extra>",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.available,
            name="Available power",
            line={"color": AVAILABLE},
            hovertemplate="<b>Available power</b>: %{y} kW<extra></extra>",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=[0] * len(data.index),
            name="Zero",
            line={"color": ZERO_LINE, "dash": "dash"},
            showlegend=False,
            hoverinfo="skip",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.charge,
            name="Charge",
            line={"color": CHARGE},
            opacity=0.9,
            hovertemplate="<b>Charge</b>: %{y} kW<extra></extra>",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.discharge,
            name="Discharge",
            line={"color": DISCHARGE, "shape": "hv"},
            opacity=0.9,
            hovertemplate="<b>Discharge</b>: %{y} kW<extra></extra>",
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

    _apply_common_layout(fig)
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
                x=[x_labels, [column] * len(x_labels)],
                y=data.positive[column],
                name=column,
                text=data.positive[column].round(3),
                textposition="outside",
            )
        )
    for column in data.negative.columns:
        fig.add_trace(
            go.Bar(
                x=[x_labels, [column] * len(x_labels)],
                y=data.negative[column],
                name=column,
                opacity=0.7,
                text=data.negative[column].round(3),
                textposition="outside",
            )
        )

    _apply_common_layout(fig, y_title="Energy (MWh)")
    fig.update_xaxes(
        tickangle=45, showdividers=False, dividercolor="rgba(0,0,0,0)", automargin=True
    )
    return fig, data.months
