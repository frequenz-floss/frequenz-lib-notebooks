# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Functions for analyzing microgrid component state transitions and extracting alerts."""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, TypeVar, cast

from frequenz.client.common.metrics import Metric
from frequenz.client.common.microgrid.electrical_components import (
    ElectricalComponentDiagnosticCode,
    ElectricalComponentStateCode,
)
from frequenz.client.common.proto import enum_from_proto
from frequenz.client.reporting import ReportingApiClient
from frequenz.client.reporting._types import MetricSample

from ._state_records import StateRecord

_logger = logging.getLogger(__name__)

E_co = TypeVar("E_co", bound=Enum, covariant=True)


class HasDiagnosticCode(Protocol[E_co]):
    """Protocol for error/warning values that have a diagnostic code."""

    diagnostic_code: int


# pylint: disable-next=too-many-arguments
async def fetch_and_extract_state_durations(
    *,
    client: ReportingApiClient,
    microgrid_components: list[tuple[int, list[int]]],
    metrics: list[Metric],
    start_time: datetime,
    end_time: datetime,
    resampling_period: timedelta | None,
    alert_states: list[ElectricalComponentStateCode],
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
        alert_states: List of ElectricalComponentStateCode that should trigger
            an alert.
        include_warnings: Whether to include warning states in the alert records.

    Returns:
        A tuple containing two lists of StateRecord instances:
            1. All state records representing the state changes.
            2. Only the alert records that match the specified alert states and
                warning inclusion criteria.
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
            Note: MetricSample.value Upstream annotation is too narrow (`float`
            only); actual values may be float | int | objects. Therefore, we need
            to cast to the expected types to satisfy the type checker.

    Returns:
        A list of StateRecord instances representing the state changes and
        error/warning durations (if any).
    """
    state_samples = sorted(samples_by_metric["state"], key=lambda s: s.timestamp)
    error_by_ts = {s.timestamp: s for s in samples_by_metric.get("error", [])}
    warning_by_ts = {s.timestamp: s for s in samples_by_metric.get("warning", [])}

    records: list[StateRecord] = []
    state_val: int | None = None
    error_val: HasDiagnosticCode[Any] | None = None
    warning_val: HasDiagnosticCode[Any] | None = None
    state_start = error_start = warning_start = None

    def emit(
        metric: str,
        val: int,
        start: datetime | None,
        end: datetime | None,
        enum_class: type[
            ElectricalComponentStateCode | ElectricalComponentDiagnosticCode
        ],
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
                emit("state", state_val, state_start, ts, ElectricalComponentStateCode)
            state_val = cast(int, sample.value)
            state_start = ts

            # Close error/warning if exiting ERROR
            if state_val != ElectricalComponentStateCode.ERROR.value:
                if error_val is not None:
                    emit(
                        "error",
                        error_val.diagnostic_code,
                        error_start,
                        ts,
                        ElectricalComponentDiagnosticCode,
                    )
                    error_val = error_start = None
                if warning_val is not None:
                    emit(
                        "warning",
                        warning_val.diagnostic_code,
                        warning_start,
                        ts,
                        ElectricalComponentDiagnosticCode,
                    )
                    warning_val = warning_start = None

        # While in ERROR
        if state_val == ElectricalComponentStateCode.ERROR.value:
            if ts in error_by_ts:
                new_err = cast(HasDiagnosticCode[Any], error_by_ts[ts].value)
                if new_err != error_val:
                    if error_val is not None:
                        emit(
                            "error",
                            error_val.diagnostic_code,
                            error_start,
                            ts,
                            ElectricalComponentDiagnosticCode,
                        )
                    error_val = new_err
                    error_start = ts

            if ts in warning_by_ts:
                new_warn = cast(HasDiagnosticCode[Any], warning_by_ts[ts].value)
                if new_warn != warning_val:
                    if warning_val is not None:
                        emit(
                            "warning",
                            warning_val.diagnostic_code,
                            warning_start,
                            ts,
                            ElectricalComponentDiagnosticCode,
                        )
                    warning_val = new_warn
                    warning_start = ts

    if state_val is not None:
        emit("state", state_val, state_start, None, ElectricalComponentStateCode)
    if state_val == ElectricalComponentStateCode.ERROR.value:
        if error_val is not None:
            emit(
                "error",
                error_val.diagnostic_code,
                error_start,
                None,
                ElectricalComponentDiagnosticCode,
            )
        if warning_val is not None:
            emit(
                "warning",
                warning_val.diagnostic_code,
                warning_start,
                None,
                ElectricalComponentDiagnosticCode,
            )
    return records


def _group_samples_by_component(
    samples: list[MetricSample], include_warnings: bool
) -> dict[tuple[int, str], dict[str, list[MetricSample]]]:
    """Group samples by (microgrid_id, component_id) and metric.

    Args:
        samples: List of MetricSample instances containing the reporting data.
        include_warnings: Whether to include warning states in the alert records.

    Returns:
        A nested dictionary where the first key is a tuple of
        (microgrid_id, component_id), and the value is another dictionary with
        keys "state", "error", optionally "warning", mapping to lists of
        MetricSample instances.
    """
    alert_metrics = {"state", "error"}
    if include_warnings:
        alert_metrics.add("warning")

    component_groups: dict[tuple[int, str], dict[str, list[MetricSample]]] = {}
    for sample in samples:
        if sample.metric not in alert_metrics:
            continue
        key = (sample.microgrid_id, str(sample.component_id))
        metric_dict = component_groups.setdefault(key, {})
        metric_dict.setdefault(sample.metric, []).append(sample)
    return component_groups


def _resolve_enum_name(
    value: int,
    enum_class: type[ElectricalComponentStateCode | ElectricalComponentDiagnosticCode],
) -> str:
    """Resolve the name of an enum member.

    Args:
        value: The integer value of the enum member to resolve.
        enum_class: The enum class to convert the value to.

    Returns:
        The name of the enum member corresponding to the given value.
    """
    result = enum_from_proto(value, enum_class, allow_invalid=False)
    return result.name


def _filter_alerts(
    all_states: list[StateRecord],
    alert_states: list[ElectricalComponentStateCode],
    include_warnings: bool,
) -> list[StateRecord]:
    """Identify alert records from all states.

    Args:
        all_states: List of all state records.
        alert_states: List of ElectricalComponentStateCode that should trigger
            an alert.
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
