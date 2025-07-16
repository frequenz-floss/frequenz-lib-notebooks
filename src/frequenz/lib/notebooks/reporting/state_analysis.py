# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Functions for analyzing microgrid component state transitions and extracting alerts."""
import logging
from datetime import datetime, timedelta

from frequenz.client.common.metric import Metric
from frequenz.client.common.microgrid.components import (
    ComponentErrorCode,
    ComponentStateCode,
)
from frequenz.client.reporting import ReportingApiClient
from frequenz.client.reporting._types import MetricSample

from ._state_records import StateRecord

_logger = logging.getLogger(__name__)


# pylint: disable-next=too-many-arguments
async def fetch_and_extract_state_durations(
    *,
    client: ReportingApiClient,
    microgrid_components: list[tuple[int, list[int]]],
    metrics: list[Metric],
    start_time: datetime,
    end_time: datetime,
    resampling_period: timedelta | None,
    alert_states: list[ComponentStateCode],
    include_warnings: bool = True,
) -> tuple[list[StateRecord], list[StateRecord]]:
    """Fetch data using the Reporting API and extract state durations and alert records.

    Args:
        client: The client used to fetch the metric samples from the Reporting API.
        microgrid_components: List of tuples where each tuple contains microgrid
            ID and corresponding component IDs.
        metrics: List of metric names.
            NOTE: The service will support requesting states without metrics in
            the future and this argument will be removed.
        start_time: The start date and time for the period.
        end_time: The end date and time for the period.
        resampling_period: The period for resampling the data. If None, data
            will be returned in its original resolution.
        alert_states: List of ComponentStateCode names that should trigger an alert.
        include_warnings: Whether to include warning states in the alert records.

    Returns:
        A tuple containing:
            - A list of StateRecord instances representing the state changes.
            - A list of StateRecord instances that match the alert criteria.
    """
    samples = await _fetch_component_data(
        client=client,
        microgrid_components=microgrid_components,
        metrics=metrics,
        start_time=start_time,
        end_time=end_time,
        resampling_period=resampling_period,
        include_states=True,
        include_bounds=False,
    )

    all_states = _extract_state_records(samples, include_warnings)
    alert_records = _filter_alerts(all_states, alert_states, include_warnings)
    return all_states, alert_records


def _extract_state_records(
    samples: list[MetricSample], include_warnings: bool
) -> list[StateRecord]:
    """Extract state records from the provided samples.

    Args:
        samples: List of MetricSample instances containing the reporting data.
        include_warnings: Whether to include warning states in the alert records.

    Returns:
        A list of StateRecord instances representing the state changes.
    """
    component_groups = _group_samples_by_component(samples, include_warnings)

    all_records = []
    for (mid, cid), metrics in component_groups.items():
        if "state" not in metrics:
            continue
        all_records.extend(_process_sample_group(mid, cid, metrics))

    all_records.sort(key=lambda x: (x.microgrid_id, x.component_id, x.start_time))
    return all_records


# pylint: disable-next=too-many-locals,too-many-branches
def _process_sample_group(
    microgrid_id: int,
    component_id: str,
    samples_by_metric: dict[str, list[MetricSample]],
) -> list[StateRecord]:
    """Process state/error/warning samples for a single component.

    Args:
        microgrid_id: ID of the microgrid.
        component_id: ID of the component.
        samples_by_metric: Dict with keys "state", "error", optionally "warning".

    Returns:
        A list of StateRecord instances representing the state changes and
        error/warning durations (if any).
    """
    state_samples = sorted(samples_by_metric["state"], key=lambda s: s.timestamp)
    error_by_ts = {s.timestamp: s for s in samples_by_metric.get("error", [])}
    warning_by_ts = {s.timestamp: s for s in samples_by_metric.get("warning", [])}

    records: list[StateRecord] = []
    state_val = error_val = warning_val = None
    state_start = error_start = warning_start = None

    def emit(
        metric: str,
        val: float,
        start: datetime | None,
        end: datetime | None,
        enum_class: type[ComponentStateCode | ComponentErrorCode],
    ) -> None:
        """Emit a state record."""
        records.append(
            StateRecord(
                microgrid_id=microgrid_id,
                component_id=component_id,
                state_type=metric,
                state_value=_resolve_enum_name(val, enum_class),
                start_time=start,
                end_time=end,
            )
        )

    for sample in state_samples:
        ts = sample.timestamp

        # State change
        if sample.value != state_val:
            if state_val is not None:
                emit("state", state_val, state_start, ts, ComponentStateCode)
            state_val = sample.value
            state_start = ts

            # Close error/warning if exiting ERROR
            if state_val != ComponentStateCode.ERROR.value:
                if error_val is not None:
                    emit("error", error_val, error_start, ts, ComponentErrorCode)
                    error_val = error_start = None
                if warning_val is not None:
                    emit("warning", warning_val, warning_start, ts, ComponentErrorCode)
                    warning_val = warning_start = None

        # While in ERROR
        if state_val == ComponentStateCode.ERROR.value:
            if ts in error_by_ts:
                new_err = error_by_ts[ts].value
                if new_err != error_val:
                    if error_val is not None:
                        emit("error", error_val, error_start, ts, ComponentErrorCode)
                    error_val = new_err
                    error_start = ts

            if ts in warning_by_ts:
                new_warn = warning_by_ts[ts].value
                if new_warn != warning_val:
                    if warning_val is not None:
                        emit(
                            "warning",
                            warning_val,
                            warning_start,
                            ts,
                            ComponentErrorCode,
                        )
                    warning_val = new_warn
                    warning_start = ts

    if state_val is not None:
        emit("state", state_val, state_start, None, ComponentStateCode)
    if state_val == ComponentStateCode.ERROR.value:
        if error_val is not None:
            emit("error", error_val, error_start, None, ComponentErrorCode)
        if warning_val is not None:
            emit("warning", warning_val, warning_start, None, ComponentErrorCode)
    return records


def _group_samples_by_component(
    samples: list[MetricSample], include_warnings: bool
) -> dict[tuple[int, str], dict[str, list[MetricSample]]]:
    """Group samples by (microgrid_id, component_id) and metric.

    Args:
        samples: List of MetricSample instances containing the reporting data.
        include_warnings: Whether to include warning states in the alert records.

    Returns:
        A dictionary where keys are tuples of (microgrid_id, component_id) and values
        are dictionaries with metric names as keys and lists of MetricSample as values.
    """
    alert_metrics = {"state", "error"}
    if include_warnings:
        alert_metrics.add("warning")

    component_groups: dict[tuple[int, str], dict[str, list[MetricSample]]] = {}
    for sample in samples:
        if sample.metric not in alert_metrics:
            continue
        key = (sample.microgrid_id, sample.component_id)
        metric_dict = component_groups.setdefault(key, {})
        metric_dict.setdefault(sample.metric, []).append(sample)
    return component_groups


def _resolve_enum_name(
    value: float, enum_class: type[ComponentStateCode | ComponentErrorCode]
) -> str:
    """Resolve the name of an enum member from its integer value.

    Args:
        value: The integer value of the enum.
        enum_class: The enum class to convert the value to.

    Returns:
        The name of the enum member if it exists, otherwise if the value is invalid,
        the enum class will return a default value (e.g., "UNSPECIFIED").
    """
    result = enum_class.from_proto(int(value))  # type: ignore[arg-type]
    return result.name


def _filter_alerts(
    all_states: list[StateRecord],
    alert_states: list[ComponentStateCode],
    include_warnings: bool,
) -> list[StateRecord]:
    """Identify alert records from all states.

    Args:
        all_states: List of all state records.
        alert_states: List of ComponentStateCode names that should trigger an alert.
        include_warnings: Whether to include warning states in the alert records.

    Returns:
        A list of StateRecord instances that match the alert criteria.
    """
    alert_metrics = ["warning", "error"] if include_warnings else ["error"]
    _alert_state_names = {state.name for state in alert_states}
    return [
        state
        for state in all_states
        if (
            (state.state_type == "state" and state.state_value in _alert_state_names)
            or (state.state_type in alert_metrics)
        )
    ]


# pylint: disable-next=too-many-arguments
async def _fetch_component_data(
    *,
    client: ReportingApiClient,
    microgrid_components: list[tuple[int, list[int]]],
    metrics: list[Metric],
    start_time: datetime,
    end_time: datetime,
    resampling_period: timedelta | None,
    include_states: bool = False,
    include_bounds: bool = False,
) -> list[MetricSample]:
    """Fetch component data from the Reporting API.

    Args:
        client: The client used to fetch the metric samples from the Reporting API.
        microgrid_components: List of tuples where each tuple contains
            microgrid ID and corresponding component IDs.
        metrics: List of metric names.
        start_time: The start date and time for the period.
        end_time: The end date and time for the period.
        resampling_period: The period for resampling the data. If None, data
            will be returned in its original resolution
        include_states: Whether to include the state data.
        include_bounds: Whether to include the bound data.

    Returns:
        List of MetricSample instances containing the reporting data.
    """
    return [
        sample
        async for sample in client.receive_microgrid_components_data(
            microgrid_components=microgrid_components,
            metrics=metrics,
            start_time=start_time,
            end_time=end_time,
            resampling_period=resampling_period,
            include_states=include_states,
            include_bounds=include_bounds,
        )
    ]
