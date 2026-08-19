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

    _DEFAULT_POWER_METRICS = {
        "power": "AC_POWER_ACTIVE",
        "energy_consumed": "AC_ENERGY_ACTIVE_CONSUMED",
        "energy_delivered": "AC_ENERGY_ACTIVE_DELIVERED",
    }

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
    async def fetch_metric_data(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = ("grid", "pv", "wind", "battery", "chp"),
        resampling_period: timedelta = timedelta(seconds=10),
        metric: str = "AC_POWER_ACTIVE",
        keep_components: bool = False,
        splits: bool = False,
    ) -> pd.DataFrame | None:
        """Fetch aggregated data for an arbitrary metric across component types.

        Args:
            microgrid_id: Microgrid ID.
            start: Start timestamp.
            end: End timestamp.
            component_types: Component types whose aggregation formulas should be
                queried for the requested metric.
            resampling_period: Sampling period used for the reporting query.
            metric: Reporting metric name to fetch.
            keep_components: Whether to include individual component IDs alongside
                the aggregated component-type columns.
            splits: Whether to append positive and negative split columns for each
                returned column.

        Returns:
            A dataframe indexed by timestamp, with one column per aggregated
            component type and, when ``keep_components`` is enabled, additional
            columns for individual component IDs. Returns ``None`` when no data is
            available.
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

        dup_mask = df.duplicated(keep="first")
        if not dup_mask.empty:
            _logger.info("Found %s rows that have duplicates", dup_mask.sum())
        df = df[~dup_mask]

        df = df.pivot_table(index="timestamp", columns="component_id", values="value")
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
            for cid in all_cids:
                if cid not in df.columns:
                    _logger.warning(
                        "Component ID %s not found in data, setting zero", cid
                    )
                    df.loc[:, cid] = np.nan

        df.columns = [str(e) for e in df.columns]  # type: ignore

        if splits:
            df = self._split_positive_negative(df)

        ctypes = list(rename_cols.values())
        new_cols = [e for e in ctypes if e in df.columns] + sorted(
            [e for e in df.columns if e not in ctypes]
        )
        return df[new_cols]

    # pylint: disable=too-many-locals
    async def metric_data(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = (
            "grid",
            "pv",
            "battery",
            "wind",
            "chp",
            "consumption",
        ),
        resampling_period: timedelta = timedelta(seconds=10),
        metric: str | None = None,
        metrics: dict[str, str] | None = None,
        keep_components: bool = False,
        splits: bool = False,
        data_fetch_mode: Literal["power", "energy"] = "power",
    ) -> pd.DataFrame | None:
        """Return component-type power data, fetched directly or derived from energy.

        Args:
            microgrid_id: Microgrid ID.
            start: Start timestamp.
            end: End timestamp.
            component_types: List of component types to be aggregated.
            resampling_period: Data resampling period.
            metric: Power metric used only in ``"power"`` mode. This is kept for
                compatibility with the previous generic metric fetcher API.
            metrics: Metrics used for fetching data. In ``"power"`` mode this
                includes the power metric. In ``"energy"`` mode this includes the
                consumed and delivered cumulative energy metrics that are netted
                and converted to power.
            keep_components: Include individual components in output.
            splits: Include columns for positive and negative power values for components.
            data_fetch_mode: Choose whether power is fetched directly from a power
                metric or derived from net cumulative energy and then converted to
                power.

        Returns:
            DataFrame with power data of aggregated components, regardless of
            whether it was fetched directly as power or derived from net energy,
            or ``None`` if no data is available.

        Raises:
            ValueError: If ``data_fetch_mode`` is unknown, or if ``metric`` is
                passed with ``data_fetch_mode="energy"``.
        """
        if data_fetch_mode not in ("power", "energy"):
            raise ValueError(f"Unknown data_fetch_mode: {data_fetch_mode}")
        if data_fetch_mode == "energy" and metric is not None:
            raise ValueError(
                '`metric` can only be used with data_fetch_mode="power"; '
                "override energy metrics with `metrics` instead"
            )

        fetch_metrics = dict(self._DEFAULT_POWER_METRICS)
        if metrics is not None:
            fetch_metrics.update(metrics)
        if metric is not None:
            fetch_metrics["power"] = metric

        if data_fetch_mode == "power":
            return await self.fetch_metric_data(
                microgrid_id=microgrid_id,
                start=start,
                end=end,
                component_types=component_types,
                resampling_period=resampling_period,
                metric=fetch_metrics["power"],
                keep_components=keep_components,
                splits=splits,
            )

        net_energy = await self.ac_active_energy_net(
            microgrid_id=microgrid_id,
            start=start,
            end=end,
            component_types=component_types,
            resampling_period=resampling_period,
            keep_components=keep_components,
            splits=False,
            unit="Wh",
            metrics=fetch_metrics,
        )
        if net_energy is None:
            return None
        power_df = self._convert_cumulative_energy_to_power(
            net_energy, resampling_period=resampling_period
        )
        if splits:
            power_df = self._split_positive_negative(power_df)
        return power_df

    async def ac_active_power(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = ("grid", "pv", "wind", "battery", "chp"),
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
        df = await self.fetch_metric_data(
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
        df = await self.fetch_metric_data(
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
        df = await self.fetch_metric_data(
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

    async def ac_active_energy_net(  # pylint: disable=too-many-arguments
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
        metrics: dict[str, str] | None = None,
    ) -> pd.DataFrame | None:
        """Net active energy as consumed - delivered."""
        fetch_metrics = dict(self._DEFAULT_POWER_METRICS)
        if metrics is not None:
            fetch_metrics.update(metrics)

        energy_consumed, energy_delivered = await asyncio.gather(
            self.fetch_metric_data(
                microgrid_id=microgrid_id,
                start=start,
                end=end,
                component_types=component_types,
                resampling_period=resampling_period,
                metric=fetch_metrics["energy_consumed"],
                keep_components=keep_components,
                splits=False,
            ),
            self.fetch_metric_data(
                microgrid_id=microgrid_id,
                start=start,
                end=end,
                component_types=component_types,
                resampling_period=resampling_period,
                metric=fetch_metrics["energy_delivered"],
                keep_components=keep_components,
                splits=False,
            ),
        )
        if energy_consumed is None and energy_delivered is None:
            return None
        converted_consumed = (
            None
            if energy_consumed is None
            else self._convert_units(
                energy_consumed,
                unit=unit,
                scale_by_unit={"Wh": 1, "kWh": 1000, "MWh": 1e6},
            )
        )
        converted_delivered = (
            None
            if energy_delivered is None
            else self._convert_units(
                energy_delivered,
                unit=unit,
                scale_by_unit={"Wh": 1, "kWh": 1000, "MWh": 1e6},
            )
        )
        net_energy = self._net(converted_consumed, converted_delivered)
        if net_energy is None or not splits:
            return net_energy
        return self._split_positive_negative(net_energy)

    @staticmethod
    def _net(
        consumed: pd.DataFrame | None, delivered: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """Return consumed - delivered, treating a missing side as zero."""
        if consumed is None and delivered is None:
            return None
        if consumed is None:
            return -delivered
        if delivered is None:
            return consumed
        return consumed.sub(delivered, fill_value=0)

    @staticmethod
    def _convert_cumulative_energy_to_power(
        df: pd.DataFrame, *, resampling_period: timedelta
    ) -> pd.DataFrame:
        """Convert cumulative energy meter readings into power values."""
        energy_delta = df.diff().shift(-1)
        negative_deltas = energy_delta < 0
        if negative_deltas.to_numpy().any():
            _logger.warning(
                "Ignoring %s negative cumulative-energy deltas while converting "
                "to power, likely due to meter resets.",
                int(negative_deltas.sum().sum()),
            )
            energy_delta = energy_delta.mask(negative_deltas)
        return energy_delta * (timedelta(hours=1) / resampling_period)
