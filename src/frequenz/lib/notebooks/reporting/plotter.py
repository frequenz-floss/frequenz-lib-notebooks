# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Plotting functions for the reporting module."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from frequenz.lib.notebooks.reporting.utils.helpers import build_color_map, long_to_wide


# pylint: disable=too-many-arguments, too-many-positional-arguments,
# pylint: disable=use-dict-literal, too-many-locals
def plot_time_series(
    df: pd.DataFrame,
    time_col: str | None = None,
    cols: list[str] | None = None,
    title: str = "Time Series Plot",
    xaxis_title: str = "Timestamp",
    yaxis_title: str = "kW",
    legend_title: str = "Components",
    color_dict: dict[str, str] | None = None,
    long_format_flag: bool = False,
    category_col: str | None = None,
    value_col: str | None = None,
    fill_cols: list[str] | None = None,
    plot_order: list[str] | None = None,
) -> go.Figure:
    """Create an interactive time-series plot using Plotly.

    Generates a multi-line time-series plot from a DataFrame, optionally handling
    long-to-wide data transformations and area fills for selected columns. The
    plot includes zoom controls, a range slider, and a date range selector.

    Args:
        df: Input DataFrame containing time and numeric data.
        time_col: Name of the timestamp column to use as the x-axis. If None,
            the current index is used.
        cols: List of numeric columns to plot. If None, all numeric columns
            except `time_col` are plotted.
        title: Plot title displayed at the top. Defaults to "Time Series Plot".
        xaxis_title: Label for the x-axis. Defaults to "Timestamp".
        yaxis_title: Label for the y-axis. Defaults to "kW".
        legend_title: Title for the legend. Defaults to "Components".
        color_dict: Optional dictionary mapping column names to custom colors.
            If not provided, default Plotly colors are used.
        long_format_flag: Whether to convert the DataFrame from long to wide
            format before plotting. Defaults to False.
        category_col: Column name for categories when converting from long to
            wide format. Used only if `long_format_flag=True`.
        value_col: Column name for values when converting from long to wide
            format. Used only if `long_format_flag=True`.
        fill_cols: List of column names to plot as filled areas under the curve.
            Defaults to None (no fill).
        plot_order: Optional list specifying the order of columns to plot. If None,
            the order in `cols` is used.

    Returns:
        A Plotly Figure object representing the interactive time-series plot.

    Raises:
        KeyError: If `time_col` is specified but not found in the DataFrame.
    """
    # Decide which axis to use for time
    if time_col is not None:
        if time_col not in df.columns:
            raise KeyError(f"Column '{time_col}' not found in DataFrame.")
        pdf = df.set_index(time_col)
    else:
        pdf = df.copy()

    # Convert long to wide if necessary
    if long_format_flag:
        pdf = long_to_wide(
            pdf, time_col=pdf.index, category_col=category_col, value_col=value_col
        )

    # Determine which columns to plot (and in what order)
    if cols is None:
        cols = [c for c in pdf.select_dtypes(include="number").columns if c != time_col]

    # Safe reorder: use plot_order if provided, else keep cols as-is
    cols = [c for c in (plot_order or cols) if c in pdf.columns]

    # Legend ranking independent of draw order
    rank_map = {c: i for i, c in enumerate(cols)}

    # Colour Mapping
    color_map = build_color_map(cols, color_dict)

    # Timeseries-Plot
    fig = go.Figure()

    # Check if fill_cols is provided
    if fill_cols is None:
        fill_cols = []

    # Add one line trace per column
    for i, col in enumerate(cols):
        fill_mode = "tonextx" if col in fill_cols else "none"
        line_color = color_map.get(col)
        fill_color = (
            line_color.replace("1)", "0.3)") if isinstance(line_color, str) else None
        )

        fig.add_trace(
            go.Scatter(
                x=pdf.index,
                y=pdf[col],
                mode="lines",
                name=col,
                line=dict(color=line_color, shape="hv"),
                fill=fill_mode,
                fillcolor=fill_color,
                legendrank=rank_map.get(col, 10_000 + i),
                showlegend=True,
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
            rangeselector=dict(
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
        legend=dict(title=dict(text=legend_title), traceorder="normal"),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
    )
    return fig


def plot_energy_pie_chart(
    power_df: pd.DataFrame, color_dict: dict[str, str] | None = None
) -> go.Figure:
    """Create an interactive donut (pie) chart of energy sources.

    Generates a pie chart showing the relative energy contributions from
    different sources (e.g., PV, grid, CHP), with percentage labels and
    hover details in kilowatt-hours.

    Args:
        power_df: DataFrame containing at least two columns:
            - `"Energiebezug"`: Category or energy source name.
            - `"Energie [kWh]"`: Corresponding energy values.
        color_dict: Optional dictionary mapping energy sources (Energiebezug)
            to custom color hex codes or rgba strings. If not provided,
            Plotly's default color sequence is used.

    Returns:
        A Plotly Figure object representing a donut-style energy distribution chart.
    """
    fig = px.pie(
        power_df,
        names="Energiebezug",
        values="Energie [kWh]",
        hole=0.4,
        color="Energiebezug",
        color_discrete_map=color_dict or {},
    )

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
