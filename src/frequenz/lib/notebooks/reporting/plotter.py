# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Plotting functions for the reporting module."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_overview_line_chart(overview_df: pd.DataFrame) -> go.Figure:
    """Plot an overview line chart of all components over time."""
    fig = go.Figure()

    for col in overview_df.columns:
        if col != "Zeitpunkt":
            fig.add_trace(
                go.Scatter(
                    x=overview_df["Zeitpunkt"],
                    y=overview_df[col],
                    mode="lines",
                    name=col,
                )
            )

    fig.update_layout(
        title="Lastgang Übersicht",
        xaxis_title="Zeitpunkt",
        yaxis_title="kW",
        xaxis={"tickformat": "%Y-%m-%d %H:%M", "title_font": {"size": 14}},
        legend={"title": "Komponenten"},
        template="plotly_white",
    )
    fig.show()


def plot_energy_pie_chart(power_df: pd.DataFrame) -> px.pie:
    """Plot a pie chart for energy source contributions."""
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
    fig.show()


def plot_pv_analysis(pv_analyse_df: pd.DataFrame) -> go.Figure:
    """Plot PV Einspeisung and Netzanschluss lines if available."""
    fig = go.Figure()

    for label in ["PV Einspeisung", "Netzanschluss"]:
        if label in pv_analyse_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pv_analyse_df["Zeitpunkt"],
                    y=pv_analyse_df[label],
                    mode="lines",
                    name=label,
                )
            )

    fig.update_layout(
        title="PV Leistung",
        xaxis_title="Zeitpunkt",
        yaxis_title="kW",
        xaxis={"tickformat": "%Y-%m-%d %H:%M", "title_font": {"size": 14}},
        legend={"title": "Komponenten"},
        template="plotly_white",
    )
    fig.show()


def plot_battery_throughput(bat_analyse_df: pd.DataFrame) -> go.Figure:
    """Plot battery throughput over time."""
    fig = go.Figure()

    if "Batterie Durchsatz" in bat_analyse_df.columns:
        fig.add_trace(
            go.Scatter(
                x=bat_analyse_df["Zeitpunkt"],
                y=bat_analyse_df["Batterie Durchsatz"],
                mode="lines",
                name="Batterie Durchsatz",
            )
        )

    fig.update_layout(
        title="Batterie Durchsatz",
        xaxis_title="Zeitpunkt",
        yaxis_title="kW",
        xaxis={"tickformat": "%Y-%m-%d %H:%M", "title_font": {"size": 14}},
        template="plotly_white",
    )
    fig.show()


def plot_grid_import(master_df: pd.DataFrame) -> go.Figure:
    """Plot Netzbezug over time if the column exists."""
    fig = go.Figure()

    if "Netzbezug" in master_df.columns:
        fig.add_trace(
            go.Scatter(
                x=master_df["Zeitpunkt"],
                y=master_df["Netzbezug"],
                mode="lines",
                name="Netzbezug",
            )
        )

    fig.update_layout(
        title="Netzbezug",
        xaxis_title="Zeitpunkt",
        yaxis_title="kW",
        xaxis={"tickformat": "%Y-%m-%d %H:%M", "title_font": {"size": 14}},
        template="plotly_white",
    )
    fig.show()
