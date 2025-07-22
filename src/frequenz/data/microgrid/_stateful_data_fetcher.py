# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH
"""A helper module to write data exports.

This class is a wrapper around the `MicrogridData` class to add state
designed for continuous data exports. It not only receives data, but also
writes the latest data point to an on disc buffer. The timestamp in the
buffer is then used as the start time for the next run.

Furthermore this module introduces a transactional approach to updating
the on disc buffer to ensure data integrity:

- Buffer updates are written to a temporary file (e.g.,
data.parquet.tmp).
- On success: The temporary file is atomically renamed to its final
path, committing the new buffer state.
- On failure: The temporary file is deleted, rolling back the change.

This ensures the unprocessed data will be fetched again on the next
run.

???+ example

    ```python
    REPORTING_API_KEY = "your_api_key"
    REPORTING_API_SECRET = "your_api_secret"

    mid = 12345
    component_name = "battery"
    metric = "active_power"

    mg_config = MicrogridConfig.load_configs(
        microgrid_config_dir=Path("/path/to/microgrid/configs")
    )
    mg_data = MicrogridData(
        "https://reporting.example.com",
        REPORTING_API_KEY,
        REPORTING_API_SECRET,
        mg_config,
    )

    data_fetcher = StatefulDataFetcher(
        microgrid_data,
        data_buffer_dir=Path("/path/to/data/buffer"),
        resampling_period=timedelta(seconds=180),
    ),

    try:
        data = await data_fetcher.receive_microgrid_data(
            mid,
            component_name,
            metric,
        )
        _logger.debug(
            "Data fetched for %s: %s", location.component.name, data
        )
    except ValueError as e:
        _logger.error("Error fetching data for AKS ID %s: %s", aks_id, e)

    # do sth with data
    # ...

    data_fetcher.commit()  # Commit the changes if successful
    # data_fetcher.rollback()  # Rollback if there was an error
    ```
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .component_data import MicrogridData

_logger = logging.getLogger(__name__)

# Number of data points to keep in the buffer. We only look at the last 1 data
# point but can be increased if needed for debugging or testing purposes.
BUFFER_SIZE = 1

# The reporting API implementation follows an eventually consistent system design
# and therefore we use a 15-minute delta to get a high probability that the
# received data is stabilized.
END_TIME_DELTA = timedelta(minutes=15)


class StatefulDataFetcher:
    """A helper class to handle fetching of microgrid component data.

    This class provides methods to query new data for a specific microgrid and
    its components, and to write the updated data to a temporary file.
    The file is written only temporarily and needs to be committed or rolled back
    in case of success or failure, respectively.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        microgrid_data: MicrogridData,
        data_buffer_dir: Path,
        resampling_period: timedelta,
        end_time_delta: timedelta = END_TIME_DELTA,
        initial_period: timedelta = timedelta(hours=1),
    ) -> None:
        """Initialize the TransactionalDataExport class.

        Args:
            microgrid_data: An instance of MicrogridData to query data from.
            data_buffer_dir: The path to the directory for buffering data.
            resampling_period: The period to resample the data.
            end_time_delta: The time subtracted from now to be passed as
                            end_time to the API.
            initial_period: The initial period to use for the first data fetch.
        """
        _logger.debug("Initializing StatefulDataFetcher instance.")
        self._microgrid_data = microgrid_data
        self._data_buffer = data_buffer_dir
        self._resampling_period = resampling_period
        self._end_time_delta = end_time_delta
        self._initial_period = initial_period

        self._data_buffer.mkdir(parents=True, exist_ok=True)

        # Maps temp_path -> final_path for files that need to be committed
        self._temp_files_to_commit: dict[Path, Path] = {}

    @staticmethod
    def _generate_filename(
        microgrid_id: int, components: tuple[str, ...], metric: str
    ) -> str:
        """Generate a unique, sanitized filename.

        Args:
            microgrid_id: The ID of the microgrid.
            components: The component types to include in the filename.
            metric: The name of the metric to include in the filename.

        Returns:
            A sanitized string suitable for use as a filename.
        """
        sorted_components = sorted([c.lower().replace(" ", "_") for c in components])
        return (
            f"mid{microgrid_id}_{'_'.join(sorted_components)}_{metric.lower()}.parquet"
        )

    async def receive_microgrid_data(
        self,
        microgrid_id: int,
        components: tuple[str, ...],
        metric: str,
    ) -> pd.DataFrame | None:
        """Query new microgrid data and write the updated buffer to a temp file.

        Reads the last timestamp from the main buffer file, queries for new data,
        and writes the combined data to a new temporary buffer file.

        Args:
            microgrid_id: The ID of the microgrid to query.
            components: The component types to include in the query.
            metric: The metric to query.

        Returns:
            A pandas DataFrame with only the new data points, or None.
        """
        _logger.debug(
            "Processing data for microgrid ID %s, components %s",
            microgrid_id,
            components,
        )

        final_parquet_file = self._data_buffer / self._generate_filename(
            microgrid_id, components, metric
        )
        temp_parquet_file = final_parquet_file.with_suffix(
            final_parquet_file.suffix + ".tmp"
        )

        now = datetime.now(timezone.utc)
        end_time = now - self._end_time_delta
        on_disk_buffer = pd.DataFrame()
        start_time: datetime

        try:
            on_disk_buffer = pd.read_parquet(final_parquet_file)
            if not on_disk_buffer.empty and isinstance(
                on_disk_buffer.index, pd.DatetimeIndex
            ):
                last_timestamp = on_disk_buffer.index.max().to_pydatetime()
                start_time = last_timestamp + timedelta(microseconds=1)
                _logger.info("Read buffer. Last timestamp: %s.", last_timestamp)
            else:
                _logger.info(
                    "Buffer file is empty. Defaulting to: %s.", self._initial_period
                )
                start_time = now - self._initial_period
        except FileNotFoundError:
            _logger.info(
                "No buffer file at '%s'. Defaulting to %s.",
                final_parquet_file,
                self._initial_period,
            )
            start_time = now - self._initial_period

        if start_time >= end_time:
            _logger.info("Buffer is up to date. No new data to fetch.")
            return None

        _logger.info("Fetching new data from %s to %s...", start_time, end_time)

        new_df = await self._microgrid_data.metric_data(
            microgrid_id=microgrid_id,
            start=start_time,
            end=end_time,
            component_types=components,
            resampling_period=self._resampling_period,
            metric=metric,
        )

        if new_df is not None and not new_df.empty:
            _logger.info("Fetched %d new data points.", len(new_df))
            combined_df = pd.concat([on_disk_buffer, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
            combined_df.sort_index(inplace=True)
            updated_buffer = combined_df.tail(BUFFER_SIZE)

            # Write to temporary file instead of final destination
            updated_buffer.to_parquet(temp_parquet_file, engine="pyarrow")

            _logger.info(
                "Wrote updated buffer with %d points to temporary file '%s'.",
                len(updated_buffer),
                temp_parquet_file,
            )

            self._temp_files_to_commit[temp_parquet_file] = final_parquet_file
        else:
            _logger.info("No new data was returned from the API.")

        return new_df

    def commit(self) -> None:
        """Commit the temporary files to their final locations."""
        _logger.info("Reports sent successfully. Committing buffer updates.")
        for temp_path, final_path in self._temp_files_to_commit.items():
            try:
                os.replace(temp_path, final_path)
                _logger.debug(
                    "Committed buffer update: %s -> %s", temp_path, final_path
                )
            except OSError as e:
                _logger.error(
                    "Failed to commit buffer update from %s to %s: %s",
                    temp_path,
                    final_path,
                    e,
                )

        self._temp_files_to_commit.clear()

    def rollback(self) -> None:
        """Rollback the temporary files if the transaction fails."""
        for temp_path in self._temp_files_to_commit:
            try:
                temp_path.unlink()
                _logger.debug("Rolled back by deleting temp file: %s", temp_path)
            except OSError as unlink_e:
                _logger.error(
                    "Failed to clean up temporary buffer file %s during rollback: %s",
                    temp_path,
                    unlink_e,
                )

        self._temp_files_to_commit.clear()
