# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Fetch component type power data from the reporting service."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from frequenz.client.common.metric import Metric
from frequenz.client.reporting import ReportingApiClient

from .config import MicrogridConfig

_logger = logging.getLogger(__name__)


class MicrogridData:
    """Fetch power data for component types of a microgrid."""

    def __init__(self, server_url: str, key: str, microgrid_config_path: str) -> None:
        """Initialize microgrid data.

        Args:
            server_url: URL of the reporting service.
            key: Authentication key to the service.
            microgrid_config_path: Path to the config file with microgrid components.
        """
        self._client = ReportingApiClient(server_url=server_url, key=key)

        self._microgrid_configs = MicrogridConfig.load_configs(microgrid_config_path)

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

    # pylint: disable=too-many-locals
    async def metric_data(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = ("grid", "pv", "battery"),
        resampling_period: timedelta = timedelta(seconds=10),
        metric: str = "AC_ACTIVE_POWER",
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

        metric_enum = Metric[metric.upper()]
        data = [
            sample
            for ctype, formula in formulas.items()
            async for sample in self._client.receive_aggregated_data(
                microgrid_id=microgrid_id,
                metric=metric_enum,
                aggregation_formula=formula,
                start=start,
                end=end,
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
                async for sample in self._client.list_microgrid_components_data(
                    microgrid_components=microgrid_components,
                    metrics=metric_enum,
                    start_dt=start,
                    end_dt=end,
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

        cols = df.columns
        if splits:
            pos_cols = [f"{col}_pos" for col in cols]
            neg_cols = [f"{col}_neg" for col in cols]
            df[pos_cols] = df[cols].clip(lower=0)
            df[neg_cols] = df[cols].clip(upper=0)

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
            metric="AC_ACTIVE_POWER",
            keep_components=keep_components,
            splits=splits,
        )
        if df is None:
            return df

        if unit == "W":
            pass
        if unit == "kW":
            df = df / 1000
        elif unit == "MW":
            df = df / 1e6
        else:
            raise ValueError(f"Unknown unit: {unit}")
        return df

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
