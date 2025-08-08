# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Plotly visualization for asset optimization reporting."""


from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .viz_mpl import require_columns


def _split_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end), …] for each contiguous True-run in *mask*."""
    if not mask.any():
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return list(zip(starts, ends))

def _stepify(
    x: pd.DatetimeIndex,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (x, y) into explicit hv-step representation:
    value y[i] applies from x[i] to x[i+1]
    """
    if len(x) < 2:
        return x.to_numpy(), y

    x2 = np.repeat(x.to_numpy(), 2)[1:]
    y2 = np.repeat(y, 2)[:-1]
    return x2, y2

# pylint: disable=too-many-arguments, too-many-positional-arguments
def _add_fill(
    fig: go.Figure,
    idx: pd.DatetimeIndex,
    upper: np.ndarray,
    lower: np.ndarray,
    segments: list[tuple[int, int]],
    rgba: str,
    name: str,
) -> None:
    """Add step-aligned filled polygons (hv semantics)."""

    for s, e in segments:
        # slice incl. endpoint
        x_seg = idx[s : e + 1]
        up_seg = upper[s : e + 1]
        lo_seg = lower[s : e + 1]

        # step-expand both boundaries
        x_up, y_up = _stepify(x_seg, up_seg)
        x_lo, y_lo = _stepify(x_seg, lo_seg)

        # closed polygon
        xs = np.concatenate([x_up, x_lo[::-1]])
        ys = np.concatenate([y_up, y_lo[::-1]])

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"width": 0},
                fill="toself",
                fillcolor=rgba,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # legend proxy
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 10, "color": rgba},
            name=name,
            hoverinfo="skip",
        )
    )


def plot_power_flow(df: pd.DataFrame, show: bool = False) -> go.Figure:
    """Interactive Plotly clone of the Matplotlib power-flow plot (step-wise)."""
    d = -df.copy()
    idx = cast(pd.DatetimeIndex, d.index)
    cons = -d["consumption"].to_numpy()

    chp = d["chp"].to_numpy() if "chp" in d else np.zeros_like(cons)
    pv = d["pv"].clip(lower=0).to_numpy() if "pv" in d else np.zeros_like(cons)
    bat = d["battery"].to_numpy() if "battery" in d else None
    grid = -d["grid"].to_numpy() if "grid" in d else None

    fig = go.Figure()

    # CHP base
    if "chp" in d:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=chp,
                name="CHP",
                mode="lines",
                line={"width": 0},
                line_shape="hv",
                fill="tozeroy",
                fillcolor="rgba(100,149,237,0.50)",
                hoverinfo="skip",
            )
        )

    # PV stacked on CHP
    if "pv" in d:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=chp + pv,
                name="PV (on CHP)" if "chp" in d else "PV",
                mode="lines",
                line={"width": 0},
                line_shape="hv",
                fill="tonexty",
                fillcolor="rgba(255,215,0,0.70)",
                hoverinfo="skip",
            )
        )

    # Battery polygons
    if bat is not None:
        bat_cons = -(d["consumption"] + d["battery"]).to_numpy()
        _add_fill(
            fig,
            idx,
            bat_cons,
            cons,
            _split_segments(bat_cons > cons),
            "rgba(0,128,0,0.30)",
            "Charge",
        )
        _add_fill(
            fig,
            idx,
            cons,
            bat_cons,
            _split_segments(bat_cons < cons),
            "rgba(240,128,128,0.55)",
            "Discharge",
        )

    # Grid line
    if grid is not None:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=grid,
                name="Grid",
                mode="lines",
                line={"color": "grey"},
                line_shape="hv",
            )
        )

    # Consumption line
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=cons,
            name="Consumption",
            mode="lines",
            line={"color": "black", "width": 2},
            line_shape="hv",
        )
    )

    # Layout (unchanged)
    fig.update_layout(
        title="Microgrid Power Flow",
        xaxis_title="Time",
        yaxis_title="Power [kW]",
        hovermode="x unified",
        template="plotly_white",
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        legend={"orientation": "v", "x": 1.01, "y": 1},
        height=600,
        autosize=True,
        width=None,
    )
    fig.update_yaxes(range=[min(0, cons.min()), None])

    if show:
        fig.show()
    return fig


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return list(zip(starts, ends))


# pylint: disable=too-many-arguments, too-many-positional-arguments
def _add_polygon(
    fig: go.Figure,
    x: pd.DatetimeIndex,
    upper: np.ndarray,
    lower: np.ndarray,
    segments: list[tuple[int, int]],
    rgba: str,
    name: str,
) -> None:
    for s, e in segments:
        xs = np.concatenate([x[s : e + 1], x[s : e + 1][::-1]])
        ys = np.concatenate([upper[s : e + 1], lower[s : e + 1][::-1]])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"width": 0},
                line_shape="hv",
                fill="toself",
                fillcolor=rgba,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 10, "color": rgba},
            name=name,
            hoverinfo="skip",
        )
    )


def plot_battery_power(df: pd.DataFrame, show: bool = False) -> go.Figure:
    """Battery power & SoC with step-wise lines, all else identical."""
    require_columns(df, "soc", "battery", "grid")

    idx = cast(pd.DatetimeIndex, df.index)
    soc = df["soc"]
    assert isinstance(soc, pd.Series), "soc must be a pandas Series"

    bat = df["battery"].to_numpy()
    avail = (df["battery"] - df["grid"]).to_numpy()

    max_abs = max(map(abs, (bat.min(), bat.max(), avail.min(), avail.max()))) * 1.1

    fig = go.Figure()

    # SoC grey band (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=soc,
            mode="lines",
            line={"width": 0},
            line_shape="hv",
            fill="tozeroy",
            fillcolor="rgba(128,128,128,0.40)",
            yaxis="y2",
            name="SOC",
        )
    )

    # Charge / Discharge polygons
    _add_polygon(
        fig,
        idx,
        bat,
        np.zeros_like(bat),
        _contiguous_runs(bat > 0),
        "rgba(0,128,0,0.90)",
        "Charge",
    )
    _add_polygon(
        fig,
        idx,
        np.zeros_like(bat),
        bat,
        _contiguous_runs(bat < 0),
        "rgba(255,0,0,0.90)",
        "Discharge",
    )

    # Available-power black line
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=avail,
            mode="lines",
            line={"color": "black"},
            line_shape="hv",
            name="Available power",
        )
    )

    # Zero reference
    fig.add_shape(
        type="line",
        x0=idx.min(),
        x1=idx.max(),
        y0=0,
        y1=0,
        line={"color": "grey", "dash": "dash"},
        yref="y",
        xref="x",
    )

    # Layout (unchanged except for title note)
    fig.update_layout(
        title="Battery Power & State of Charge",
        xaxis={"title": "Time"},
        yaxis={"title": "Battery Power [kW]", "range": [-max_abs, max_abs]},
        yaxis2={
            "title": "Battery SOC [%]",
            "range": [0, 100],
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 60, "t": 60, "b": 40},
    )

    if show:
        fig.show()

    fig.update_layout(
        height=600, autosize=True, width=None
    )  # Plotly fills the container

    return fig
