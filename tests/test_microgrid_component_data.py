# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for grid energy handling in MicrogridData.ac_active_power."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from frequenz.data.microgrid import MicrogridData

_START = datetime(2026, 1, 9, 6, 30, tzinfo=timezone.utc)
_END = datetime(2026, 1, 9, 6, 45, tzinfo=timezone.utc)


class _Stub(MicrogridData):
    """MicrogridData with fixed power and directional grid-energy frames."""

    def __init__(  # pylint: disable=super-init-not-called
        self, consumed: list[float] | None = None
    ) -> None:
        self.metric_calls: list[dict[str, object]] = []
        self.energy_calls: list[dict[str, object]] = []
        self._consumed = consumed or [1.0, 1.0, float("nan")]

    async def metric_data(self, **kwargs: object) -> pd.DataFrame | None:
        """Return signed grid power with splits, plus PV, in W."""
        self.metric_calls.append(kwargs)
        return pd.DataFrame(
            {
                "grid": [3000.0] * 3,
                "pv": [-2000.0] * 3,
                "grid_pos": [3000.0] * 3,
                "grid_neg": [0.0] * 3,
            }
        )

    async def ac_active_energy(self, **kwargs: object) -> pd.DataFrame | None:
        """Return directional grid energy deltas, NaN in the final period."""
        self.energy_calls.append(kwargs)
        values = (
            self._consumed
            if kwargs["direction"] == "consumed"
            else [0.5, 0.5, float("nan")]
        )
        return pd.DataFrame({"grid": values})


async def _power(mdata: _Stub, **kwargs: object) -> pd.DataFrame | None:
    """Fetch active power from the stub for a fixed interval."""
    return await mdata.ac_active_power(
        microgrid_id=241,
        start=_START,
        end=_END,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_ac_active_power_defaults_to_grid_power() -> None:
    """By default the signed grid power is returned and no energy is queried."""
    mdata = _Stub()
    result = await _power(mdata)

    assert not mdata.energy_calls
    assert result is not None
    assert result["grid"].tolist() == [3.0] * 3
    assert len(result) == 3


async def test_ac_active_power_energy_replaces_grid_power() -> None:
    """Energy mode swaps the grid power family for directional energy columns."""
    mdata = _Stub()
    result = await _power(mdata, grid_data_source="energy", splits=True)

    assert mdata.metric_calls[0]["splits"] is True
    assert [call["component_types"] for call in mdata.energy_calls] == [
        ("grid",),
        ("grid",),
    ]
    assert [call["direction"] for call in mdata.energy_calls] == [
        "consumed",
        "delivered",
    ]
    assert result is not None
    assert result.columns.tolist() == [
        "pv",
        "grid_energy_consumed",
        "grid_energy_delivered",
    ]
    assert result["pv"].tolist() == [-2.0, -2.0]
    # The final period has no forward energy delta and is dropped.
    assert len(result) == 2
    assert not result.isna().any().any()


async def test_ac_active_power_energy_warns_on_counter_decrease(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A decreasing cumulative counter is reported to the user."""
    mdata = _Stub(consumed=[-5.0, 1.0, float("nan")])
    with caplog.at_level("WARNING"):
        await _power(mdata, grid_data_source="energy")

    assert "consumed energy counter decreased in 1 reporting periods" in caplog.text


async def test_ac_active_power_rejects_unknown_grid_data_source() -> None:
    """Unknown sources fail before any query is sent."""
    mdata = _Stub()
    with pytest.raises(ValueError, match="Unknown grid data source"):
        await _power(mdata, grid_data_source="bogus")

    assert not mdata.metric_calls
