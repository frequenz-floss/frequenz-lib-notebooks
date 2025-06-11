import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ───────────────────────── helpers ──────────────────────────
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


def _add_fill(fig, idx, upper, lower, segments, rgba, name):
    """One closed polygon per (start,end) segment."""
    for s, e in segments:
        xs = np.concatenate([idx[s : e + 1], idx[s : e + 1][::-1]])
        ys = np.concatenate([upper[s : e + 1], lower[s : e + 1][::-1]])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=0),
                line_shape="hv",  # ← step edge
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
            marker=dict(size=10, color=rgba),
            name=name,
            hoverinfo="skip",
        )
    )


# ─────────────────────── power-flow plot ─────────────────────
def plot_power_flow2(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Interactive Plotly clone of the Matplotlib power-flow plot (step-wise)."""
    d = -df.copy()
    idx = d.index
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
                line=dict(width=0),
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
                line=dict(width=0),
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
                line=dict(color="grey"),
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
            line=dict(color="black", width=2),
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
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="v", x=1.01, y=1),
        height=800,
        autosize=True,
        width=None,
    )
    fig.update_yaxes(range=[min(0, cons.min()), None])

    if show:
        fig.show()
    return fig


# ───────────────────────── second helpers ──────────────────────
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


def _add_polygon(fig, x, upper, lower, segments, rgba, name):
    for s, e in segments:
        xs = np.concatenate([x[s : e + 1], x[s : e + 1][::-1]])
        ys = np.concatenate([upper[s : e + 1], lower[s : e + 1][::-1]])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=0),
                line_shape="hv",  # ← step edge
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
            marker=dict(size=10, color=rgba),
            name=name,
            hoverinfo="skip",
        )
    )


# ─────────────────── battery-power plot ────────────────────────
def plot_battery_power2(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Battery power & SoC with step-wise lines, all else identical."""
    if "soc" not in df.columns or df["soc"].nunique() <= 1:
        raise ValueError("df must contain an 'soc' column that varies")

    idx = df.index
    soc = df["soc"].iloc[:, 0] if df["soc"].ndim > 1 else df["soc"]
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
            line=dict(width=0),
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
            line=dict(color="black"),
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
        line=dict(color="grey", dash="dash"),
        yref="y",
        xref="x",
    )

    # Layout (unchanged except for title note)
    fig.update_layout(
        title="Battery Power & State of Charge",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Battery Power [kW]", range=[-max_abs, max_abs]),
        yaxis2=dict(
            title="Battery SOC [%]",
            range=[0, 100],
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=60, t=60, b=40),
    )

    if show:
        fig.show()

    fig.update_layout(
        height=600, autosize=True, width=None
    )  # Plotly fills the container

    return fig
