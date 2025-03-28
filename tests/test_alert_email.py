# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for the frequenz.lib.notebooks.alerts module."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pytest

from frequenz.lib.notebooks.alerts.alert_email import (
    AlertPlotType,
    ExportOptions,
    ImageExportFormat,
    _coerce_enum,
    _coerce_formats,
    plot_alerts,
)


@pytest.fixture
def sample_alerts_df() -> pd.DataFrame:
    """Create a sample DataFrame with dummy alert records."""
    return pd.DataFrame(
        {
            "microgrid_id": ["A", "A", "B"],
            "component_id": ["inv1", "inv2", "inv1"],
            "state_value": [1, 2, 1],
            "start_time": pd.date_range("2023-01-01", periods=3, freq="h"),
        }
    )


def test_enum_coercion_accepts_str_and_enum() -> None:
    """_coerce_enum accepts both str and Enum values."""
    assert _coerce_enum("summary", AlertPlotType, "plot_type") == AlertPlotType.SUMMARY
    assert (
        _coerce_enum(AlertPlotType.ALL, AlertPlotType, "plot_type") == AlertPlotType.ALL
    )


def test_enum_coercion_raises_on_invalid() -> None:
    """_coerce_enum raises ValueError for invalid string values."""
    with pytest.raises(ValueError):
        _coerce_enum("invalid", AlertPlotType, "plot_type")


def test_format_coercion_from_str_and_list() -> None:
    """_coerce_formats converts single or multiple string/enum formats into a list of enums."""
    assert _coerce_formats("html") == [ImageExportFormat.HTML]
    assert _coerce_formats(["html", "png"]) == [
        ImageExportFormat.HTML,
        ImageExportFormat.PNG,
    ]


def test_plot_alerts_display_only(sample_alerts_df: pd.DataFrame) -> None:
    """plot_alerts should only display plots when export format is None and show=True."""
    with patch("plotly.graph_objs._figure.Figure.show") as mock_show:
        result = plot_alerts(
            sample_alerts_df,
            plot_type="all",
            export_options=ExportOptions(format=None, show=True),
        )
        assert result is None
        expected_count = sum(
            1 for p in AlertPlotType if p not in (AlertPlotType.ALL)
        )  # One for each plot
        assert mock_show.call_count == expected_count


def test_plot_alerts_no_output(sample_alerts_df: pd.DataFrame) -> None:
    """plot_alerts should do nothing when export format is None and show=False."""
    with patch("plotly.graph_objs._figure.Figure.show") as mock_show:
        result = plot_alerts(
            sample_alerts_df, export_options=ExportOptions(format=None, show=False)
        )
        assert result is None
        mock_show.assert_not_called()


@pytest.mark.parametrize(
    "formats",
    ["html"],
)
def test_plot_alerts_export(
    formats: str | list[str | ImageExportFormat], sample_alerts_df: pd.DataFrame
) -> None:
    """plot_alerts should save plots to one or more formats."""
    with TemporaryDirectory() as tmpdir:
        opts = ExportOptions(format=formats, output_dir=tmpdir, show=False)
        paths = plot_alerts(sample_alerts_df, plot_type="summary", export_options=opts)
        assert isinstance(paths, list)
        num_formats = len(formats) if isinstance(formats, list) else 1
        assert len(paths) == num_formats * 1
        for path in paths:
            assert Path(path).exists()


def test_plot_alerts_empty_df_does_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """plot_alerts should log and skip processing for empty DataFrames."""
    caplog.set_level("DEBUG")
    df = pd.DataFrame()
    result = plot_alerts(df)
    assert result is None
    assert any("no plots generated" in msg.lower() for msg in caplog.text.splitlines())
