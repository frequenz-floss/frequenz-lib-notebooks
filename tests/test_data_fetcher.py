# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for the frequenz.data.microgrid._stateful_data_fetcher module."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from frequenz.data.microgrid import StatefulDataFetcher


@pytest.fixture
def mock_microgrid_data() -> MagicMock:
    """Provide a mock for the MicrogridData dependency."""
    return MagicMock()


@pytest.fixture
def data_fetcher(tmp_path: Path, mock_microgrid_data: MagicMock) -> StatefulDataFetcher:
    """Provide an instance of StatefulDataFetcher with a temporary buffer directory."""
    return StatefulDataFetcher(
        microgrid_data=mock_microgrid_data,
        data_buffer_dir=tmp_path,
        resampling_period=timedelta(seconds=360),
    )


class TestStatefulDataFetcher:
    """Unit tests for the StatefulDataFetcher class."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test that the data buffer directory is created on initialization."""
        assert not (tmp_path / "data").exists()
        StatefulDataFetcher(
            microgrid_data=MagicMock(),
            data_buffer_dir=tmp_path / "data",
            resampling_period=timedelta(seconds=360),
        )
        assert (tmp_path / "data").exists()

    def test_generate_filename(self, data_fetcher: StatefulDataFetcher) -> None:
        """Test the filename generation logic."""
        filename = data_fetcher._generate_filename(  # pylint: disable=protected-access
            microgrid_id=123, components=("Battery", "Grid Meter"), metric="Power"
        )
        # Components should be sorted alphabetically and lowercased
        assert filename == "mid123_battery_grid_meter_power.parquet"

    @patch("frequenz.data.microgrid._stateful_data_fetcher.datetime")
    async def test_receive_data_no_initial_file(
        self,
        mock_datetime: MagicMock,
        data_fetcher: StatefulDataFetcher,
        mock_microgrid_data: MagicMock,
    ) -> None:
        """Test fetching data when no previous buffer file exists."""
        # pylint: disable=protected-access
        # --- Setup ---
        fixed_now = datetime(2025, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_now

        start_time = fixed_now - timedelta(hours=1)
        end_time = fixed_now - data_fetcher._end_time_delta

        new_data = pd.DataFrame(
            {"value": [1.0, 2.0]},
            index=pd.to_datetime(
                [
                    start_time + timedelta(minutes=1),
                    start_time + timedelta(minutes=2),
                ],
                utc=True,
            ),
        )
        mock_microgrid_data.metric_data = AsyncMock(return_value=new_data)

        # --- Execution ---
        result_df = await data_fetcher.receive_microgrid_data(
            microgrid_id=1, components=("Battery",), metric="SoC"
        )

        # --- Assertions ---
        mock_microgrid_data.metric_data.assert_awaited_once_with(
            microgrid_id=1,
            start=start_time,
            end=end_time,
            component_types=("Battery",),
            resampling_period=data_fetcher._resampling_period,
            metric="SoC",
        )

        # Check that a temp file was created and is pending commit
        assert len(data_fetcher._temp_files_to_commit) == 1
        temp_file = list(data_fetcher._temp_files_to_commit.keys())[0]
        assert temp_file.name.endswith(".parquet.tmp")

        # Check the content of the written file (tailed to BUFFER_SIZE=1)
        written_data = pd.read_parquet(temp_file)
        assert len(written_data) == 1
        pd.testing.assert_frame_equal(written_data, new_data.tail(1))

        # Check that the returned dataframe is the complete new data
        assert result_df is not None
        pd.testing.assert_frame_equal(result_df, new_data)

    @patch("frequenz.data.microgrid._stateful_data_fetcher.datetime")
    async def test_receive_data_with_existing_file(
        self,
        mock_datetime: MagicMock,
        data_fetcher: StatefulDataFetcher,
        mock_microgrid_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test fetching data when a buffer file already exists."""
        # pylint: disable=protected-access
        # --- Setup ---
        fixed_now = datetime(2025, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_now

        # Create an existing buffer file
        last_timestamp = fixed_now - timedelta(hours=1)
        on_disk_buffer = pd.DataFrame(
            {"value": [100.0]}, index=pd.to_datetime([last_timestamp], utc=True)
        )
        final_file_path = tmp_path / "mid2_inverter_power.parquet"
        on_disk_buffer.to_parquet(final_file_path)

        # This is the new data the API will return
        new_data_start_time = last_timestamp + timedelta(minutes=3)
        new_data = pd.DataFrame(
            {"value": [101.0]}, index=pd.to_datetime([new_data_start_time], utc=True)
        )
        mock_microgrid_data.metric_data = AsyncMock(return_value=new_data)

        # --- Execution ---
        await data_fetcher.receive_microgrid_data(
            microgrid_id=2, components=("Inverter",), metric="Power"
        )

        # --- Assertions ---
        # Start time should be based on the last timestamp from the file
        expected_start_time = last_timestamp + timedelta(microseconds=1)
        mock_microgrid_data.metric_data.assert_awaited_once()
        call_args = mock_microgrid_data.metric_data.call_args[1]
        assert call_args["start"] == expected_start_time

        # A temp file should be pending commit
        assert len(data_fetcher._temp_files_to_commit) == 1
        temp_file = list(data_fetcher._temp_files_to_commit.keys())[0]

        # The new buffer should contain the latest point
        written_data = pd.read_parquet(temp_file)
        assert len(written_data) == 1
        pd.testing.assert_frame_equal(written_data, new_data)

    @patch("frequenz.data.microgrid._stateful_data_fetcher.datetime")
    async def test_receive_data_up_to_date(
        self,
        mock_datetime: MagicMock,
        data_fetcher: StatefulDataFetcher,
        mock_microgrid_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that no data is fetched if the buffer is already up to date."""
        # pylint: disable=protected-access
        # --- Setup ---
        fixed_now = datetime(2025, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_now

        # Last timestamp is within the END_TIME_DELTA, so we are "up-to-date"
        last_timestamp = fixed_now - data_fetcher._end_time_delta + timedelta(minutes=1)
        on_disk_buffer = pd.DataFrame(
            {"value": [50.0]}, index=pd.to_datetime([last_timestamp], utc=True)
        )
        final_file_path = tmp_path / "mid3_meter_power.parquet"
        on_disk_buffer.to_parquet(final_file_path)

        mock_microgrid_data.metric_data = AsyncMock()

        # --- Execution ---
        result = await data_fetcher.receive_microgrid_data(
            microgrid_id=3, components=("Meter",), metric="Power"
        )

        # --- Assertions ---
        assert result is None
        mock_microgrid_data.metric_data.assert_not_awaited()
        assert not data_fetcher._temp_files_to_commit

    async def test_receive_data_api_returns_nothing(
        self, data_fetcher: StatefulDataFetcher, mock_microgrid_data: MagicMock
    ) -> None:
        """Test behavior when the data API returns None or an empty DataFrame."""
        # pylint: disable=protected-access
        mock_microgrid_data.metric_data = AsyncMock(return_value=pd.DataFrame())

        result = await data_fetcher.receive_microgrid_data(
            microgrid_id=4, components=("EV Charger",), metric="Current"
        )

        assert isinstance(result, pd.DataFrame) and result.empty
        mock_microgrid_data.metric_data.assert_awaited_once()
        assert not data_fetcher._temp_files_to_commit  # No temp file should be created

    @patch("frequenz.data.microgrid._stateful_data_fetcher.os.replace")
    def test_commit(
        self, mock_os_replace: MagicMock, data_fetcher: StatefulDataFetcher
    ) -> None:
        """Test the commit functionality."""
        # pylint: disable=protected-access
        temp_path = Path("/tmp/file.tmp")
        final_path = Path("/tmp/file.final")
        data_fetcher._temp_files_to_commit = {temp_path: final_path}

        data_fetcher.commit()

        mock_os_replace.assert_called_once_with(temp_path, final_path)

    @patch("pathlib.Path.unlink")
    def test_rollback(
        self, mock_unlink: MagicMock, data_fetcher: StatefulDataFetcher
    ) -> None:
        """Test the rollback functionality."""
        # pylint: disable=protected-access
        temp_path = Path("/tmp/file.tmp")
        final_path = Path("/tmp/file.final")
        data_fetcher._temp_files_to_commit = {temp_path: final_path}

        data_fetcher.rollback()

        mock_unlink.assert_called_once()
