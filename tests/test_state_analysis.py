# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for the frequenz.reporting package."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import pytest
from frequenz.client.common.microgrid.electrical_components import (
    ElectricalComponentDiagnosticCode,
    ElectricalComponentStateCode,
)
from frequenz.client.common.proto import enum_from_proto
from frequenz.client.reporting._types import MetricSample

from frequenz.lib.notebooks.reporting.state_analysis import (
    _extract_state_records,
    _filter_alerts,
    _resolve_enum_name,
)


@dataclass(frozen=True)
class _DiagnosticValue:
    """Test helper to mimic reporting diagnostic payloads."""

    diagnostic_code: int


test_cases_extract_state_durations = [
    {
        "description": "Empty samples",
        "samples": [],
        "alert_states": [enum_from_proto(1, ElectricalComponentStateCode)],
        "include_warnings": True,
        "expected_all_states": [],
        "expected_alert_records": [],
    },
    {
        "description": "No matching metrics",
        "samples": [
            MetricSample(datetime(2023, 1, 1, 0, 0), 1, "101", "temperature", 25),
            MetricSample(datetime(2023, 1, 1, 1, 0), 1, "101", "humidity", 60),
        ],
        "alert_states": [enum_from_proto(1, ElectricalComponentStateCode)],
        "include_warnings": True,
        "expected_all_states": [],
        "expected_alert_records": [],
    },
    {
        "description": "Single state change",
        "samples": [
            MetricSample(datetime(2023, 1, 1, 0, 0), 1, "101", "state", 0),
            MetricSample(datetime(2023, 1, 1, 1, 0), 1, "101", "state", 1),
        ],
        "alert_states": [enum_from_proto(1, ElectricalComponentStateCode)],
        "include_warnings": True,
        "expected_all_states": [
            {
                "microgrid_id": 1,
                "component_id": "101",
                "state_type": "state",
                "state_value": _resolve_enum_name(0, ElectricalComponentStateCode),
                "start_time": datetime(2023, 1, 1, 0, 0),
                "end_time": datetime(2023, 1, 1, 1, 0),
            },
            {
                "microgrid_id": 1,
                "component_id": "101",
                "state_type": "state",
                "state_value": _resolve_enum_name(1, ElectricalComponentStateCode),
                "start_time": datetime(2023, 1, 1, 1, 0),
                "end_time": None,
            },
        ],
        "expected_alert_records": [
            {
                "microgrid_id": 1,
                "component_id": "101",
                "state_type": "state",
                "state_value": _resolve_enum_name(1, ElectricalComponentStateCode),
                "start_time": datetime(2023, 1, 1, 1, 0),
                "end_time": None,
            },
        ],
    },
    {
        "description": "Warnings and errors included",
        "samples": [
            MetricSample(datetime(2023, 1, 2, 0, 0), 3, "303", "state", 0),
            MetricSample(datetime(2023, 1, 2, 0, 30), 3, "303", "state", 10),
            MetricSample(
                datetime(2023, 1, 2, 0, 30),
                3,
                "303",
                "warning",
                cast(Any, _DiagnosticValue(10)),
            ),
            MetricSample(datetime(2023, 1, 2, 1, 0), 3, "303", "state", 1),
            MetricSample(datetime(2023, 1, 2, 1, 30), 3, "303", "state", 20),
            MetricSample(
                datetime(2023, 1, 2, 1, 30),
                3,
                "303",
                "error",
                cast(Any, _DiagnosticValue(20)),
            ),
        ],
        "alert_states": [enum_from_proto(1, ElectricalComponentStateCode)],
        "include_warnings": True,
        "expected_all_states": [
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "state",
                "state_value": _resolve_enum_name(0, ElectricalComponentStateCode),
                "start_time": datetime(2023, 1, 2, 0, 0),
                "end_time": datetime(2023, 1, 2, 0, 30),
            },
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "state",
                "state_value": _resolve_enum_name(10, ElectricalComponentStateCode),
                "start_time": datetime(2023, 1, 2, 0, 30),
                "end_time": datetime(2023, 1, 2, 1, 0),
            },
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "state",
                "state_value": _resolve_enum_name(1, ElectricalComponentStateCode),
                "start_time": datetime(2023, 1, 2, 1, 0),
                "end_time": datetime(2023, 1, 2, 1, 30),
            },
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "state",
                "state_value": _resolve_enum_name(20, ElectricalComponentStateCode),
                "start_time": datetime(2023, 1, 2, 1, 30),
                "end_time": None,
            },
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "warning",
                "state_value": _resolve_enum_name(
                    10, ElectricalComponentDiagnosticCode
                ),
                "start_time": datetime(2023, 1, 2, 0, 30),
                "end_time": datetime(2023, 1, 2, 1, 0),
            },
        ],
        "expected_alert_records": [
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "warning",
                "state_value": _resolve_enum_name(
                    10, ElectricalComponentDiagnosticCode
                ),
                "start_time": datetime(2023, 1, 2, 0, 30),
                "end_time": datetime(2023, 1, 2, 1, 0),
            },
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "state",
                "state_value": _resolve_enum_name(1, ElectricalComponentStateCode),
                "start_time": datetime(2023, 1, 2, 1, 0),
                "end_time": datetime(2023, 1, 2, 1, 30),
            },
        ],
    },
]


@pytest.mark.parametrize(
    "test_case", test_cases_extract_state_durations, ids=lambda tc: tc["description"]
)
def test_extract_and_filter_state_records(test_case: dict[str, Any]) -> None:
    """Test extracting and filtering state records from samples."""
    _all_states = _extract_state_records(
        test_case["samples"], test_case["include_warnings"]
    )
    _alert_records = _filter_alerts(
        _all_states,
        alert_states=test_case["alert_states"],
        include_warnings=test_case["include_warnings"],
    )
    all_states = [record._asdict() for record in _all_states]
    alert_records = [record._asdict() for record in _alert_records]

    expected_all_states = test_case["expected_all_states"]
    expected_alert_records = test_case["expected_alert_records"]

    all_states_sorted = sorted(
        all_states,
        key=lambda x: (
            x["microgrid_id"],
            x["component_id"],
            x["state_type"],
            x["start_time"],
        ),
    )
    expected_all_states_sorted = sorted(
        expected_all_states,
        key=lambda x: (
            x["microgrid_id"],
            x["component_id"],
            x["state_type"],
            x["start_time"],
        ),
    )

    alert_records_sorted = sorted(
        alert_records,
        key=lambda x: (
            x["microgrid_id"],
            x["component_id"],
            x["state_type"],
            x["start_time"],
        ),
    )
    expected_alert_records_sorted = sorted(
        expected_alert_records,
        key=lambda x: (
            x["microgrid_id"],
            x["component_id"],
            x["state_type"],
            x["start_time"],
        ),
    )
    assert all_states_sorted == expected_all_states_sorted
    assert alert_records_sorted == expected_alert_records_sorted


@pytest.mark.parametrize(
    "value, enum_class, expected_name",
    [
        (
            ElectricalComponentStateCode.READY.value,
            ElectricalComponentStateCode,
            "READY",
        ),
        (ElectricalComponentStateCode.OFF.value, ElectricalComponentStateCode, "OFF"),
        (
            ElectricalComponentDiagnosticCode.OVERTEMPERATURE.value,
            ElectricalComponentDiagnosticCode,
            "OVERTEMPERATURE",
        ),
        (
            ElectricalComponentDiagnosticCode.SHORT_CIRCUIT.value,
            ElectricalComponentDiagnosticCode,
            "SHORT_CIRCUIT",
        ),
    ],
)
def test_resolve_enum_name(value: int, enum_class: Any, expected_name: str) -> None:
    """Test resolving enum names from integer values."""
    result = _resolve_enum_name(value, enum_class)
    assert result == expected_name
