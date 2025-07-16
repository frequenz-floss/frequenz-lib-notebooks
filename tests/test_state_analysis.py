# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for the frequenz.reporting package."""
from datetime import datetime
from typing import Any

import pytest
from frequenz.client.common.microgrid.components import (
    ComponentErrorCode,
    ComponentStateCode,
)
from frequenz.client.reporting._types import MetricSample

from frequenz.lib.notebooks.reporting.state_analysis import (
    _extract_state_records,
    _filter_alerts,
    _resolve_enum_name,
)

test_cases_extract_state_durations = [
    {
        "description": "Empty samples",
        "samples": [],
        "alert_states": [ComponentStateCode.from_proto(1)],  # type: ignore[arg-type]
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
        "alert_states": [ComponentStateCode.from_proto(1)],  # type: ignore[arg-type]
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
        "alert_states": [ComponentStateCode.from_proto(1)],  # type: ignore[arg-type]
        "include_warnings": True,
        "expected_all_states": [
            {
                "microgrid_id": 1,
                "component_id": "101",
                "state_type": "state",
                "state_value": _resolve_enum_name(0, ComponentStateCode),
                "start_time": datetime(2023, 1, 1, 0, 0),
                "end_time": datetime(2023, 1, 1, 1, 0),
            },
            {
                "microgrid_id": 1,
                "component_id": "101",
                "state_type": "state",
                "state_value": _resolve_enum_name(1, ComponentStateCode),
                "start_time": datetime(2023, 1, 1, 1, 0),
                "end_time": None,
            },
        ],
        "expected_alert_records": [
            {
                "microgrid_id": 1,
                "component_id": "101",
                "state_type": "state",
                "state_value": _resolve_enum_name(1, ComponentStateCode),
                "start_time": datetime(2023, 1, 1, 1, 0),
                "end_time": None,
            },
        ],
    },
    {
        "description": "Warnings and errors included",
        "samples": [
            MetricSample(datetime(2023, 1, 2, 0, 0), 3, "303", "state", 0),
            MetricSample(datetime(2023, 1, 2, 0, 30), 3, "303", "warning", 10),
            MetricSample(datetime(2023, 1, 2, 1, 0), 3, "303", "state", 1),
            MetricSample(datetime(2023, 1, 2, 1, 30), 3, "303", "error", 20),
        ],
        "alert_states": [ComponentStateCode.from_proto(1)],  # type: ignore[arg-type]
        "include_warnings": True,
        "expected_all_states": [
            # State transitions
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "state",
                "state_value": _resolve_enum_name(0, ComponentStateCode),
                "start_time": datetime(2023, 1, 2, 0, 0),
                "end_time": datetime(2023, 1, 2, 1, 0),
            },
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "state",
                "state_value": _resolve_enum_name(1, ComponentStateCode),
                "start_time": datetime(2023, 1, 2, 1, 0),
                "end_time": None,
            },
            # Warning transitions
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "warning",
                "state_value": _resolve_enum_name(10, ComponentErrorCode),
                "start_time": datetime(2023, 1, 2, 0, 30),
                "end_time": None,
            },
            # Error transitions
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "error",
                "state_value": _resolve_enum_name(20, ComponentErrorCode),
                "start_time": datetime(2023, 1, 2, 1, 30),
                "end_time": None,
            },
        ],
        "expected_alert_records": [
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "warning",
                "state_value": _resolve_enum_name(10, ComponentErrorCode),
                "start_time": datetime(2023, 1, 2, 0, 30),
                "end_time": None,
            },
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "error",
                "state_value": _resolve_enum_name(20, ComponentErrorCode),
                "start_time": datetime(2023, 1, 2, 1, 30),
                "end_time": None,
            },
            # State alert
            {
                "microgrid_id": 3,
                "component_id": "303",
                "state_type": "state",
                "state_value": _resolve_enum_name(1, ComponentStateCode),
                "start_time": datetime(2023, 1, 2, 1, 0),
                "end_time": None,
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
        (ComponentStateCode.READY.value, ComponentStateCode, "READY"),
        (ComponentStateCode.OFF.value, ComponentStateCode, "OFF"),
        (
            ComponentErrorCode.OVERTEMPERATURE.value,
            ComponentErrorCode,
            "OVERTEMPERATURE",
        ),
        (ComponentErrorCode.SHORT_CIRCUIT.value, ComponentErrorCode, "SHORT_CIRCUIT"),
        (9999.0, ComponentStateCode, "UNSPECIFIED"),  # Invalid state code
        (-8888.0, ComponentErrorCode, "UNSPECIFIED"),  # Invalid error code
    ],
)
def test_resolve_enum_name(value: int, enum_class: Any, expected_name: str) -> None:
    """Test resolving enum names from float values."""
    result = _resolve_enum_name(float(value), enum_class)
    assert result == expected_name
