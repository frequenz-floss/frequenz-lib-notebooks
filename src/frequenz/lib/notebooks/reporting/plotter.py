# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Plotting functions for the reporting module."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# pylint: disable=too-many-arguments, too-many-positional-arguments, disable=use-dict-literal
def plot_time_series(
    df: pd.DataFrame,
    time_col: str | None = None,
    cols: list[str] | None = None,
    title: str = "Time Series Plot",
    xaxis_title: str = "Timestamp",
    yaxis_title: str = "kW",
    legend_title: str = "Components",
    color_dict: dict[str, str] | None = None,
) -> go.Figure:
    """Plot a time series line chart with Plotly.

    Args:
        df: Input DataFrame containing a datetime column and one or more numeric columns.
        time_col: Optional column name to use for the x-axis (time). If None, the
            DataFrame index is used.
        cols: List of columns to plot. If None, all columns except `time_col` are plotted.
        title: Title of the plot.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        legend_title: Legend title.
        color_dict: Optional mapping from column name to color hex/string.
            Values override the default Dark2 palette.

    Returns:
        A Plotly ``go.Figure`` object.
    Raises:
        KeyError: If ``time_col`` is provided but does not exist in ``df``.
    """
    # Decide which axis to use for time
    if time_col is not None:
        if time_col not in df.columns:
            raise KeyError(f"Column '{time_col}' not found in DataFrame.")
        pdf = df.set_index(time_col)
    else:
        pdf = df.copy()

    # Select columns
    if cols is None:
        cols = [c for c in pdf.columns if c != time_col]
    else:
        cols = [col for col in cols if col in pdf.columns]

    # Default color palette
    colors = px.colors.qualitative.Dark2
    default_color_map = {col: colors[i % len(colors)] for i, col in enumerate(cols)}

    # Override default colors with user-provided mapping (if any)
    color_map = {}
    for col in cols:
        if color_dict and col in color_dict:
            color_map[col] = color_dict[col]
        else:
            color_map[col] = default_color_map[col]

    # Initialize an empty Plotly figure
    fig = go.Figure()

    # Add one line trace per column
    for col in cols:
        fig.add_trace(
            go.Scatter(
                x=pdf.index,
                y=pdf[col],
                mode="lines",
                name=col,
                line={"color": color_map[col]},
            )
        )

    # Update the figure layout: titles, legend, axes, and interactive controls
    fig.update_layout(
        title=dict(
            text=title,
            x=0.1,  # Center
            xanchor="left",
            yanchor="top",
            font=dict(size=22),
        ),
        margin=dict(t=120),
        xaxis=dict(
            type="date",
            rangeselector=dict(  # Add range selector buttons for time navigation
                buttons=[
                    dict(count=1, step="month", stepmode="backward", label="1M"),
                    dict(count=3, step="month", stepmode="backward", label="3M"),
                    dict(count=6, step="month", stepmode="backward", label="6M"),
                    dict(step="year", stepmode="todate", label="YTD"),
                    dict(count=1, step="year", stepmode="backward", label="1Y"),
                    dict(step="all", label="All"),
                ],
                bgcolor="rgba(0,0,0,0)",  # Transparent background
                activecolor="#2C7BE5",  # Highlight color for active button
                font=dict(size=12),
                x=0.65,
                xanchor="left",
                y=1.05,
                yanchor="top",
            ),
            rangeslider=dict(  # Add an interactive range slider below the x-axis
                visible=True,
                bgcolor="rgba(0,0,0,0.03)",
                bordercolor="rgba(0,0,0,0.25)",
                borderwidth=1,
                thickness=0.09,
            ),
        ),
        legend=dict(title=dict(text=legend_title)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
    )
    return fig


def plot_energy_pie_chart(power_df: pd.DataFrame) -> go.Figure:
    """Plot a donut pie chart for energy source contributions.

    Args:
        power_df: DataFrame with columns ``"Energiebezug"`` (labels) and
            ``"Energie [kWh]"`` (values).

    Returns:
        A Plotly ``go.Figure`` object configured as a donut pie chart.
    """
    fig = px.pie(power_df, names="Energiebezug", values="Energie [kWh]", hole=0.4)

    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}<br>%{percent} (%{value:.2f} kWh)<extra></extra>",
        showlegend=True,
    )

    fig.update_layout(
        title="Energiebezug",
        legend_title_text="Energiebezug",
        template="plotly_white",
        width=700,
        height=500,
    )
    return fig
