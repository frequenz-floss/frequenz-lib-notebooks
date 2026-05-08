# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Fetch component type power data from the reporting service."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from frequenz.client.common.metrics import Metric
from frequenz.client.reporting import ReportingApiClient
from frequenz.gridpool import MicrogridConfig

_logger = logging.getLogger(__name__)


class MicrogridData:
    """Fetch power data for component types of a microgrid."""

    def __init__(
        self,
        server_url: str,
        auth_key: str,
        sign_secret: str,
        microgrid_configs: dict[str, MicrogridConfig] | None = None,
    ) -> None:
        """Initialize microgrid data.

        Args:
            server_url: URL of the reporting service.
            auth_key: Authentication key to the service.
            sign_secret: Secret for signing requests.
            microgrid_configs: MicrogridConfig dict mapping microgrid IDs to MicrogridConfigs.
        """
        self._microgrid_configs = microgrid_configs
        self._client = ReportingApiClient(
            server_url=server_url, auth_key=auth_key, sign_secret=sign_secret
        )

    @property
    def microgrid_ids(self) -> list[str]:
        """Get the microgrid IDs.

        Returns:
            List of microgrid IDs.
        """
        return list(self._microgrid_configs.keys())

    @property
    def microgrid_configs(self) -> dict[str, MicrogridConfig]:
        """Return the microgrid configurations."""
        return self._microgrid_configs

    @staticmethod
    def _convert_units(
        df: pd.DataFrame,
        *,
        unit: str,
        scale_by_unit: dict[str, float],
    ) -> pd.DataFrame:
        """Convert a dataframe with values expressed in a base unit."""
        if unit not in scale_by_unit:
            raise ValueError(f"Unknown unit: {unit}")
        return df / scale_by_unit[unit]

    @staticmethod
    def _split_positive_negative(df: pd.DataFrame) -> pd.DataFrame:
        """Add positive and negative split columns for each column."""
        cols = df.columns
        pos_cols = [f"{col}_pos" for col in cols]
        neg_cols = [f"{col}_neg" for col in cols]
        df[pos_cols] = df[cols].clip(lower=0)
        df[neg_cols] = df[cols].clip(upper=0)
        return df

    # pylint: disable=too-many-locals
    async def metric_data(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = ("grid", "pv", "battery"),
        resampling_period: timedelta = timedelta(seconds=10),
        metric: str = "AC_POWER_ACTIVE",
        keep_components: bool = False,
        splits: bool = False,
    ) -> pd.DataFrame | None:
        """Power data for component types of a microgrid.

        Args:
            microgrid_id: Microgrid ID.
            start: Start timestamp.
            end: End timestamp.
            component_types: List of component types to be aggregated.
            resampling_period: Data resampling period.
            metric: Metric to be fetched.
            keep_components: Include individual components in output.
            splits: Include columns for positive and negative power values for components.

        Returns:
            DataFrame with power data of aggregated components
            or None if no data is available
        """
        mcfg = self._microgrid_configs[f"{microgrid_id}"]

        formulas = {
            ctype: mcfg.formula(ctype, metric.upper()) for ctype in component_types
        }

        logging.debug("Formulas: %s", formulas)

        metric_enum = Metric[metric.upper()]
        data = [
            sample
            for ctype, formula in formulas.items()
            async for sample in self._client.receive_aggregated_data(
                microgrid_id=microgrid_id,
                metric=metric_enum,
                aggregation_formula=formula,
                start_time=start,
                end_time=end,
                resampling_period=resampling_period,
            )
        ]

        all_cids = []
        if keep_components:
            all_cids = [
                cid
                for ctype in component_types
                for cid in mcfg.component_type_ids(ctype, metric=metric)
            ]
            _logger.debug("CIDs: %s", all_cids)
            microgrid_components = [
                (microgrid_id, all_cids),
            ]
            data_comp = [
                sample
                async for sample in self._client.receive_microgrid_components_data(
                    microgrid_components=microgrid_components,
                    metrics=metric_enum,
                    start_time=start,
                    end_time=end,
                    resampling_period=resampling_period,
                )
            ]
            data.extend(data_comp)

        if len(data) == 0:
            _logger.warning("No data found")
            return None

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        assert df["timestamp"].dt.tz is not None, "Timestamps are not tz-aware"

        # Remove duplicates
        dup_mask = df.duplicated(keep="first")
        if not dup_mask.empty:
            _logger.info("Found %s rows that have duplicates", dup_mask.sum())
        df = df[~dup_mask]

        # Pivot table
        df = df.pivot_table(index="timestamp", columns="component_id", values="value")
        # Rename formula columns
        rename_cols: dict[str, str] = {}
        for ctype, formula in formulas.items():
            if formula in rename_cols:
                _logger.warning(
                    "Ignoring %s since formula %s exists already for %s",
                    ctype,
                    formula,
                    rename_cols[formula],
                )
                continue
            rename_cols[formula] = ctype

        df = df.rename(columns=rename_cols)
        if keep_components:
            # Set missing columns to NaN
            for cid in all_cids:
                if cid not in df.columns:
                    _logger.warning(
                        "Component ID %s not found in data, setting zero", cid
                    )
                    df.loc[:, cid] = np.nan

        # Make string columns
        df.columns = [str(e) for e in df.columns]  # type: ignore

        if splits:
            df = self._split_positive_negative(df)

        # Sort columns
        ctypes = list(rename_cols.values())
        new_cols = [e for e in ctypes if e in df.columns] + sorted(
            [e for e in df.columns if e not in ctypes]
        )
        df = df[new_cols]

        return df

    async def ac_active_power(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = ("grid", "pv", "battery"),
        resampling_period: timedelta = timedelta(seconds=10),
        keep_components: bool = False,
        splits: bool = False,
        unit: str = "kW",
    ) -> pd.DataFrame | None:
        """Power data for component types of a microgrid."""
        df = await self.metric_data(
            microgrid_id=microgrid_id,
            start=start,
            end=end,
            component_types=component_types,
            resampling_period=resampling_period,
            metric="AC_POWER_ACTIVE",
            keep_components=keep_components,
            splits=splits,
        )
        if df is None:
            return df

        return self._convert_units(
            df, unit=unit, scale_by_unit={"W": 1, "kW": 1000, "MW": 1e6}
        )

    async def soc(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        resampling_period: timedelta = timedelta(seconds=10),
        keep_components: bool = False,
    ) -> pd.DataFrame | None:
        """Soc data for component types of a microgrid."""
        df = await self.metric_data(
            microgrid_id=microgrid_id,
            start=start,
            end=end,
            component_types=("battery",),
            resampling_period=resampling_period,
            metric="BATTERY_SOC_PCT",
            keep_components=keep_components,
        )
        return df

    async def ac_active_energy_consumed(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = ("grid",),
        resampling_period: timedelta = timedelta(seconds=10),
        keep_components: bool = False,
        splits: bool = False,
        unit: str = "Wh",
    ) -> pd.DataFrame | None:
        """Consumed active energy for component types of a microgrid."""
        df = await self.metric_data(
            microgrid_id=microgrid_id,
            start=start,
            end=end,
            component_types=component_types,
            resampling_period=resampling_period,
            metric="AC_ENERGY_ACTIVE_CONSUMED",
            keep_components=keep_components,
            splits=splits,
        )
        if df is None:
            return df
        return self._convert_units(
            df, unit=unit, scale_by_unit={"Wh": 1, "kWh": 1000, "MWh": 1e6}
        )

    async def ac_active_energy_delivered(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = ("grid", "pv", "battery"),
        resampling_period: timedelta = timedelta(seconds=10),
        keep_components: bool = False,
        splits: bool = False,
        unit: str = "Wh",
    ) -> pd.DataFrame | None:
        """Delivered active energy for component types of a microgrid."""
        df = await self.metric_data(
            microgrid_id=microgrid_id,
            start=start,
            end=end,
            component_types=component_types,
            resampling_period=resampling_period,
            metric="AC_ENERGY_ACTIVE_DELIVERED",
            keep_components=keep_components,
            splits=splits,
        )
        if df is None:
            return df
        return self._convert_units(
            df, unit=unit, scale_by_unit={"Wh": 1, "kWh": 1000, "MWh": 1e6}
        )

    async def _ac_active_energy_net(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...],
        resampling_period: timedelta,
        keep_components: bool,
        splits: bool,
        unit: str,
    ) -> pd.DataFrame | None:
        """Net active energy as consumed - delivered."""
        energy_consumed, energy_delivered = await asyncio.gather(
            self.ac_active_energy_consumed(
                microgrid_id=microgrid_id,
                component_types=component_types,
                start=start,
                end=end,
                resampling_period=resampling_period,
                keep_components=keep_components,
                splits=splits,
                unit=unit,
            ),
            self.ac_active_energy_delivered(
                microgrid_id=microgrid_id,
                component_types=component_types,
                start=start,
                end=end,
                resampling_period=resampling_period,
                keep_components=keep_components,
                splits=splits,
                unit=unit,
            ),
        )
        if energy_consumed is None and energy_delivered is None:
            return None
        if energy_consumed is None:
            return -energy_delivered
        if energy_delivered is None:
            return energy_consumed
        return energy_consumed.sub(energy_delivered, fill_value=0)

    # pylint: disable=too-many-arguments
    async def metric_with_present_component_types(
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        metric: Literal[
            "ac_active_power",
            "energy_consumed",
            "energy_delivered",
            "ac_active_energy",
        ] = "ac_active_power",
        resampling_period: timedelta = timedelta(seconds=10),
        keep_components: bool = True,
        splits: bool = True,
        unit: str = "kW",
    ) -> tuple[pd.DataFrame | None, list[str], list[str]]:
        """Fetch data stream and return active power with present/non-zero component types.

        Args:
            microgrid_id: The ID of the microgrid to fetch data for.
            start: The start datetime for the data query.
            end: The end datetime for the data query.
            metric: Select stream source:
                - ``"ac_active_power"`` -> ``ac_active_power``
                - ``"energy_consumed"`` -> ``ac_active_energy_consumed``
                - ``"energy_delivered"`` -> ``ac_active_energy_delivered``
                - ``"ac_active_energy"`` ->
                  ``ac_active_energy_consumed - ac_active_energy_delivered``
            resampling_period: Data resampling period.
            keep_components: Whether to keep all components in the result.
            splits: Include columns for positive and negative power values for components.
            unit: Power unit of the returned data (``"W"``, ``"kW"``, ``"MW"``).

        Returns:
            A tuple with:
            - DataFrame with active power values (or ``None`` when unavailable)
            - component types that are present as dataframe columns and non-zero
            - component types missing from the dataframe columns

        Raises:
            ValueError: If the requested metric is unknown.
        """
        mcfg = self._microgrid_configs[f"{microgrid_id}"]
        component_types = list(mcfg.component_types())
        seconds = resampling_period.total_seconds()
        if seconds <= 0:
            raise ValueError("resampling_period must be positive")

        def to_power(df_energy_wh: pd.DataFrame, *, do_splits: bool) -> pd.DataFrame:
            """Convert energy (Wh per sampling period) dataframe to power."""
            df_power_w = df_energy_wh * (3600.0 / seconds)
            df_power = self._convert_units(
                df_power_w, unit=unit, scale_by_unit={"W": 1, "kW": 1000, "MW": 1e6}
            )
            if do_splits:
                df_power = self._split_positive_negative(df_power)
            return df_power

        if metric == "ac_active_power":
            df = await self.ac_active_power(
                microgrid_id=microgrid_id,
                component_types=tuple(component_types),
                start=start,
                end=end,
                resampling_period=resampling_period,
                keep_components=keep_components,
                splits=splits,
                unit=unit,
            )
        elif metric == "energy_consumed":
            df_energy = await self.ac_active_energy_consumed(
                microgrid_id=microgrid_id,
                component_types=tuple(component_types),
                start=start,
                end=end,
                resampling_period=resampling_period,
                keep_components=keep_components,
                splits=False,
                unit="Wh",
            )
            df = None if df_energy is None else to_power(df_energy, do_splits=splits)
        elif metric == "energy_delivered":
            df_energy = await self.ac_active_energy_delivered(
                microgrid_id=microgrid_id,
                component_types=tuple(component_types),
                start=start,
                end=end,
                resampling_period=resampling_period,
                keep_components=keep_components,
                splits=False,
                unit="Wh",
            )
            df = None if df_energy is None else to_power(df_energy, do_splits=splits)
        elif metric == "ac_active_energy":
            df_energy = await self._ac_active_energy_net(
                microgrid_id=microgrid_id,
                component_types=tuple(component_types),
                start=start,
                end=end,
                resampling_period=resampling_period,
                keep_components=keep_components,
                splits=False,
                unit="Wh",
            )
            df = None if df_energy is None else to_power(df_energy, do_splits=splits)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if df is None:
            return None, [], component_types

        filtered = [
            ctype
            for ctype in component_types
            if ctype in df.columns
            and pd.to_numeric(df[ctype], errors="coerce").fillna(0).sum() != 0
        ]

        return df, filtered
