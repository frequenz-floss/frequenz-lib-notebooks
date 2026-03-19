# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Plotting functions for the reporting module."""

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from frequenz.lib.notebooks.reporting.utils.battery_usecase_plot import (
    _with_alpha,
    add_battery_usecase_overlay_traces,
    prepare_battery_usecase_plot,
)
from frequenz.lib.notebooks.reporting.utils.colors import (
    COLOR_DICT,
    LINE_DASH_MAP,
    generate_shades,
)
from frequenz.lib.notebooks.reporting.utils.helpers import build_color_map, long_to_wide


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Convert series to numeric, handling comma decimal strings."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    as_str = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(as_str, errors="coerce")


def _split_battery_power_flow(
    df: pd.DataFrame,
    cols: list[str],
    plot_order: list[str] | None,
    fill_cols: list[str] | None,
    dotted_cols: list[str] | None,
) -> tuple[
    pd.DataFrame, list[str], list[str] | None, list[str] | None, list[str] | None
]:
    """Split battery power flow into charge and discharge columns if present."""
    split_map = {
        "Batterie Leistungsfluss": ("Batterie Entladung", "Batterie Beladung"),
        "Battery Power Flow": ("Battery Discharge", "Battery Charge"),
    }

    def replace(
        seq: list[str] | None, target: str, repl: tuple[str, str]
    ) -> list[str] | None:
        if not seq:
            return seq
        new_seq: list[str] = []
        for item in seq:
            if item == target:
                new_seq.extend(repl)
            else:
                new_seq.append(item)
        return new_seq

    active_order = plot_order or cols
    for base, (discharge, charge) in split_map.items():
        if base not in df.columns:
            continue
        if base not in active_order:
            continue

        series = pd.to_numeric(df[base], errors="coerce")
        df = df.copy()
        df[charge] = series.clip(lower=0)
        df[discharge] = series.clip(upper=0)
        df = df.drop(columns=[base])

        cols = replace(cols, base, (discharge, charge)) or cols
        plot_order = replace(plot_order, base, (discharge, charge))
        fill_cols = replace(fill_cols, base, (discharge, charge))
        dotted_cols = replace(dotted_cols, base, (discharge, charge))

    return df, cols, plot_order, fill_cols, dotted_cols


def _apply_stack_for_production(
    df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply stacking for CHP, PV, and battery discharge series."""
    stack_labels = {
        "BHKW-Erzeugung",
        "BHKW Erzeugung",
        "CHP-Production",
        "CHP Production",
        "PV-Erzeugung",
        "PV Erzeugung",
        "PV-Production",
        "PV Production",
        "Batterie Entladung",
        "Battery Discharge",
        "Stromspeicher-entladen",
    }

    stackgroup_map: dict[str, str] = {}
    if not any(c in stack_labels for c in cols):
        return df, stackgroup_map

    df = df.copy()
    for col in cols:
        if col not in stack_labels or col not in df.columns:
            continue
        # Stack battery discharge as positive magnitude
        if col in {"Batterie Entladung", "Battery Discharge", "Stromspeicher-entladen"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").abs()
        stackgroup_map[col] = "production_stack"

    return df, stackgroup_map


# pylint: disable=too-many-arguments, too-many-positional-arguments,
# pylint: disable=use-dict-literal, too-many-locals, too-many-branches
# pylint: disable=too-many-statements
def plot_time_series(
    df: pd.DataFrame,
    time_col: str | None = None,
    cols: list[str] | None = None,
    title: str = "Time Series Plot",
    xaxis_title: str = "Timestamp",
    yaxis_title: str = "kW",
    legend_title: str | None = "",
    color_dict: dict[str, str] | None = None,
    long_format_flag: bool = False,
    category_col: str | None = None,
    value_col: str | None = None,
    fill_cols: list[str] | None = None,
    dotted_cols: list[str] | None = None,
    plot_order: list[str] | None = None,
    shade_by_category: bool = False,
    secondary_y_cols: Sequence[str] | None = None,
    secondary_y_title: str | None = None,
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
        dotted_cols: List of column names to render with dotted lines.
            Defaults to None (no dotted lines).
        plot_order: Optional list specifying the order of columns to plot. If None,
            the order in `cols` is used.
        shade_by_category: When plotting a long-format series, render all
            categories as different shades of the same base color.
        secondary_y_cols: Optional plotted columns to render on a secondary y-axis.
        secondary_y_title: Optional title for the secondary y-axis. Defaults to
            `secondary_y_cols` when not provided.

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

    pdf, cols, plot_order, fill_cols, dotted_cols = _split_battery_power_flow(
        pdf, cols, plot_order, fill_cols, dotted_cols
    )

    if secondary_y_cols:
        for col in secondary_y_cols:
            if col not in cols and col in pdf.columns:
                cols.append(col)

    # Safe reorder: use plot_order if provided, else keep cols as-is
    cols = [c for c in (plot_order or cols) if c in pdf.columns]

    secondary_cols: list[str] = list(secondary_y_cols or [])

    if secondary_cols:
        for col in secondary_cols:
            if col not in pdf.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
            if col not in cols:
                raise KeyError(
                    f"Secondary y-axis column '{col}' is not included in the plotted columns."
                )
    secondary_col_set = set(secondary_cols)

    pdf, stackgroup_map = _apply_stack_for_production(pdf, cols)

    # Legend ranking independent of draw order
    rank_map = {c: i for i, c in enumerate(cols)}

    # Colour Mapping
    if shade_by_category and long_format_flag and category_col and len(cols) > 1:
        base_color = (color_dict or {}).get(category_col) or COLOR_DICT.get(
            category_col
        )
        base_color = base_color or px.colors.qualitative.Plotly[0]
        shades = generate_shades(base_color, len(cols))
        color_map = {c: shades[i] for i, c in enumerate(cols)}
    else:
        color_map = build_color_map(cols, color_dict)

    # Timeseries-Plot
    fig = go.Figure()

    # Check if fill_cols is provided
    if fill_cols is None:
        fill_cols = []
    if dotted_cols is None:
        dotted_cols = []
    dotted_set = set(dotted_cols)
    # Add one line trace per column
    for i, col in enumerate(cols):
        stackgroup = stackgroup_map.get(col)
        if stackgroup:
            fill_mode = "tonexty"
        else:
            fill_mode = "tozeroy" if col in fill_cols else "none"
        line_color = color_map.get(col)
        if col.lower() == "da_price":
            line_color = COLOR_DICT.get("da_price", line_color)
        fill_color = _with_alpha(line_color, 0.9)
        y_values = _coerce_numeric_series(pdf[col])
        if col in secondary_col_set:
            if col.lower() == "da_price":
                trace_unit = "EUR/MWh"
            else:
                trace_unit = secondary_y_title or col
        else:
            trace_unit = yaxis_title

        fig.add_trace(
            go.Scatter(
                x=pdf.index,
                y=y_values,
                mode="lines",
                name=col,
                hovertemplate=f"<b>{col}</b>: %{{y}} {trace_unit}<extra></extra>",
                yaxis="y2" if col in secondary_col_set else "y",
                line=dict(
                    color=line_color,
                    shape="hv",
                    dash=(
                        "dot" if col in dotted_set else LINE_DASH_MAP.get(col, "solid")
                    ),
                    width=1,
                ),
                stackgroup=stackgroup,
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
            x=0.08,  # Center
            y=0.99,
            xanchor="left",
            yanchor="top",
            font=dict(size=22),
        ),
        height=700,
        width=950,
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
                x=0,
                xanchor="left",
                y=1.1,
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
        legend=dict(
            title=dict(text=legend_title),
            traceorder="normal",
            orientation="h",
            x=0.0,
            xanchor="left",
            y=1.2,
            yanchor="top",
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        template="plotly_white",
    )
    if secondary_cols:
        default_secondary_title = secondary_y_title or ", ".join(secondary_cols)
        yaxis2_updates: dict[str, object] = {
            "title": default_secondary_title,
            "anchor": "x",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        }
        fig.update_layout(
            yaxis2=yaxis2_updates,
        )
    return fig


# pylint: disable=too-many-statements
def plot_time_series_battery_usecase(
    df: pd.DataFrame,
    time_col: str | None = None,
    cols: list[str] | None = None,
    title: str = "Time Series Plot",
    xaxis_title: str = "Timestamp",
    yaxis_title: str = "kW",
    legend_title: str | None = "Components",
    color_dict: dict[str, str] | None = None,
    long_format_flag: bool = False,
    category_col: str | None = None,
    value_col: str | None = None,
    fill_cols: list[str] | None = None,
    dotted_cols: list[str] | None = None,
    plot_order: list[str] | None = None,
    shade_by_category: bool = False,
    battery_power_flow: str = "battery_power_flow",
    battery_charging: str = "battery_discharge",
    battery_discharging: str = "battery_charge",
    pv_col: str = "pv",
    grid_consumption_without_battery: str = "grid_consumption_without_battery",
    grid_consumption: str = "grid_consumption",
    secondary_y_cols: Sequence[str] | None = None,
    secondary_y_title: str | None = None,
) -> go.Figure:
    """Plot a battery-usecase time series with charge/discharge overlays.

    Builds a reporting plot for battery-usecase analysis by combining the
    standard time-series traces with dedicated filled overlays for battery
    charging and discharging between the grid-consumption baselines.

    Args:
        df: Source DataFrame containing the battery-usecase time series.
        time_col: Optional timestamp column to use as the x-axis.
        cols: Optional columns to plot.
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Primary y-axis label.
        legend_title: Legend title.
        color_dict: Optional color mapping for traces.
        long_format_flag: Whether ``df`` is in long format.
        category_col: Category column name for long-format inputs.
        value_col: Value column name for long-format inputs.
        fill_cols: Columns to render as filled traces.
        dotted_cols: Columns to render with dotted lines.
        plot_order: Optional explicit trace order.
        shade_by_category: Whether to generate color shades by category.
        battery_power_flow: Column containing battery power flow values.
        battery_charging: Column containing the battery charging series.
        battery_discharging: Column containing the battery discharging series.
        pv_col: Column containing PV production values.
        grid_consumption_without_battery: Column containing grid consumption
            without battery support.
        grid_consumption: Column containing grid consumption with battery
            support.
        secondary_y_cols: Optional columns to render on the secondary y-axis.
        secondary_y_title: Secondary y-axis label.

    Returns:
        A Plotly figure for battery-usecase analysis.
    """
    plot_df, cols, fill_cols, dotted_cols, plot_order, secondary_y_cols, color_dict = (
        prepare_battery_usecase_plot(
            df,
            cols=cols,
            fill_cols=fill_cols,
            dotted_cols=dotted_cols,
            plot_order=plot_order,
            secondary_y_cols=secondary_y_cols,
            color_dict=color_dict,
            time_col=time_col,
            battery_power_flow=battery_power_flow,
            battery_charging=battery_charging,
            battery_discharging=battery_discharging,
            pv_col=pv_col,
            grid_consumption_without_battery=grid_consumption_without_battery,
            grid_consumption=grid_consumption,
        )
    )
    fig = plot_time_series(
        plot_df,
        time_col=time_col,
        cols=cols,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        color_dict=color_dict,
        long_format_flag=long_format_flag,
        category_col=category_col,
        value_col=value_col,
        fill_cols=fill_cols,
        dotted_cols=dotted_cols,
        plot_order=plot_order,
        shade_by_category=shade_by_category,
        secondary_y_cols=secondary_y_cols,
        secondary_y_title=secondary_y_title,
    )
    fig.update_layout(
        legend=dict(y=1.28, yanchor="top"),
        margin=dict(t=145),
    )
    source_df = plot_df if time_col is None else plot_df.set_index(time_col)
    add_battery_usecase_overlay_traces(
        fig,
        source_df,
        color_dict=color_dict,
        yaxis_title=yaxis_title,
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
