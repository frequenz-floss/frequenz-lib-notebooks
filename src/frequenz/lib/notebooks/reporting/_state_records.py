# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Data structures for representing component state transitions and alerts."""

from datetime import datetime
from typing import NamedTuple


class StateRecord(NamedTuple):
    """A record of a component state change.

    A named tuple was chosen to allow safe access to the fields while keeping
    the simplicity of a tuple. This data type can be easily used to create a
    numpy array or a pandas DataFrame.
    """

    microgrid_id: int
    """The ID of the microgrid."""

    component_id: str
    """The ID of the component within the microgrid."""

    state_type: str
    """The type of the state (e.g., "state", "warning", "error")."""

    state_value: str
    """The value of the state (e.g., "ON", "OFF", "ERROR" etc.)."""

    start_time: datetime | None
    """The start time of the state change."""

    end_time: datetime | None
    """The end time of the state change."""
