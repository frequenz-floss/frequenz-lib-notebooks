# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Fetch component-type metric data from the reporting service."""

import logging
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Literal, Protocol, TypeAlias, cast

import numpy as np
import pandas as pd
from frequenz.client.assets import AssetsApiClient
from frequenz.client.assets.electrical_component import ElectricalComponentCategory
from frequenz.client.assets.metrics import Metric as AssetsMetric
from frequenz.client.common.metrics import Metric
from frequenz.client.common.microgrid import MicrogridId
from frequenz.client.reporting import ReportingApiClient
from frequenz.gridpool.config import MicrogridConfig

_logger = logging.getLogger(__name__)

_MicrogridConfigMapping: TypeAlias = (
    Mapping[int, MicrogridConfig] | Mapping[str, MicrogridConfig]
)


class _MicrogridConfigsContainer(Protocol):
    """A configuration document containing microgrids."""

    @property
    def microgrids(self) -> Mapping[int, MicrogridConfig]:
        """Return microgrid configurations by ID."""
        raise NotImplementedError


def _normalize_microgrid_configs(
    configs: _MicrogridConfigMapping | _MicrogridConfigsContainer,
) -> dict[int, MicrogridConfig]:
    """Normalize supported gridpool configuration formats."""
    raw_configs = configs if isinstance(configs, Mapping) else configs.microgrids
    return {int(microgrid_id): config for microgrid_id, config in raw_configs.items()}


class MicrogridData:
    """Fetch metric data for component types of a microgrid."""

    _AC_ACTIVE_ENERGY_METRICS = {
        "energy_net": "AC_ENERGY_ACTIVE",
        "energy_consumed": "AC_ENERGY_ACTIVE_CONSUMED",
        "energy_delivered": "AC_ENERGY_ACTIVE_DELIVERED",
    }

    def __init__(
        self,
        server_url: str,
        auth_key: str,
        sign_secret: str,
        microgrid_configs: (
            _MicrogridConfigMapping | _MicrogridConfigsContainer | None
        ) = None,
        assets_client: AssetsApiClient | None = None,
    ) -> None:
        """Initialize microgrid data.

        Args:
            server_url: URL of the reporting service.
            auth_key: Authentication key to the service.
            sign_secret: Secret for signing requests.
            microgrid_configs: A mapping of microgrid IDs to configurations, or a
                gridpool configuration document containing that mapping.
            assets_client: Optional Assets API client used to fetch static battery
                SOC rated bounds.
        """
        self._microgrid_configs = (
            None
            if microgrid_configs is None
            else _normalize_microgrid_configs(microgrid_configs)
        )
        self._client = ReportingApiClient(
            server_url=server_url, auth_key=auth_key, sign_secret=sign_secret
        )
        self._assets_client = assets_client

    @property
    def microgrid_ids(self) -> list[int]:
        """Get the microgrid IDs.

        Returns:
            List of microgrid IDs.
        """
        if self._microgrid_configs is None:
            return []
        return list(self._microgrid_configs.keys())

    @property
    def microgrid_configs(self) -> dict[int, MicrogridConfig] | None:
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
    def _add_split_pos_neg_cols(df: pd.DataFrame) -> pd.DataFrame:
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
        microgrid_id: int | str,
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
        metric: str = "AC_POWER_ACTIVE",
        keep_components: bool = False,
        splits: bool = False,
    ) -> pd.DataFrame | None:
        """Fetch aggregated data for an arbitrary metric across component types.

        Args:
            microgrid_id: Microgrid ID. Numeric strings are accepted for notebook
                compatibility.
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
            DataFrame with power data of aggregated components
            or None if no data is available

        Raises:
            ValueError: If microgrid configurations are not loaded.
        """
        if self._microgrid_configs is None:
            raise ValueError("Microgrid configurations are not loaded.")
        microgrid_id = int(microgrid_id)
        mcfg = self._microgrid_configs[microgrid_id]
        metric = metric.upper()

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
            df = self._add_split_pos_neg_cols(df)

        ctypes = list(rename_cols.values())
        new_cols = [e for e in ctypes if e in df.columns] + sorted(
            [e for e in df.columns if e not in ctypes]
        )
        return df[new_cols]

    async def ac_active_power(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int | str,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = (
            "grid",
            "pv",
            "wind",
            "battery",
            "chp",
            "consumption",
        ),
        resampling_period: timedelta = timedelta(seconds=10),
        keep_components: bool = False,
        splits: bool = False,
        unit: str = "kW",
        from_energy: bool = False,
    ) -> pd.DataFrame | None:
        """Power data for component types of a microgrid.

        Set ``from_energy`` to derive power from net AC active energy instead of
        fetching the instantaneous power metric.
        """
        if from_energy:
            df = await self.ac_active_energy(
                microgrid_id=microgrid_id,
                start=start,
                end=end,
                component_types=component_types,
                resampling_period=resampling_period,
                keep_components=keep_components,
                unit="Wh",
            )
            if df is not None:
                df = self._convert_cumulative_energy_to_power(
                    df, resampling_period=resampling_period
                )
                if splits:
                    df = self._add_split_pos_neg_cols(df)
        else:
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
        microgrid_id: int | str,
        start: datetime,
        end: datetime,
        resampling_period: timedelta = timedelta(seconds=10),
        keep_components: bool = False,
    ) -> pd.DataFrame | None:
        """Fetch and aggregate SOC for the configured battery components.

        Battery component IDs are read from the microgrid configuration.  Their
        SOC and capacity values are fetched directly from Reporting, rather
        than through formulas in the configuration. The aggregate SOC is a
        capacity-weighted average, using each component's own capacity reading
        at each timestamp so that capacity changes are reflected over time.

        Args:
            microgrid_id: The ID of the microgrid.
            start: The start time for the data fetch.
            end: The end time for the data fetch.
            resampling_period: The period for resampling the data.
            keep_components: Whether to keep individual component data.

        Returns:
            A DataFrame containing the aggregated SOC data, or None if no data is available.

        Raises:
            ValueError: If microgrid configurations are not loaded.
        """
        if self._microgrid_configs is None:
            raise ValueError("Microgrid configurations are not loaded.")

        mg_id = int(microgrid_id)
        config = self._microgrid_configs[mg_id]
        component_ids = config.component_type_ids(
            "battery", component_category="component"
        )
        if not component_ids:
            _logger.warning(
                "No battery components are configured for microgrid %s.", mg_id
            )
            return None

        df = await self._battery_metric_component_data(
            microgrid_id=mg_id,
            component_ids=component_ids,
            metric=Metric.BATTERY_SOC_PCT,
            metric_name="battery SOC",
            start=start,
            end=end,
            resampling_period=resampling_period,
        )
        if df is None:
            return None

        capacity_df = await self._battery_metric_component_data(
            microgrid_id=mg_id,
            component_ids=component_ids,
            metric=Metric.BATTERY_CAPACITY,
            metric_name="battery capacity",
            start=start,
            end=end,
            resampling_period=resampling_period,
        )
        weighted_soc = None
        if capacity_df is not None:
            capacity_aligned = self._align_forward_filled(capacity_df, df.index)
            weighted_soc = self._capacity_weighted_soc(df, capacity_aligned)
        if weighted_soc is None:
            return df if keep_components else None
        df["battery"] = weighted_soc

        if not keep_components:
            component_cols = [col for col in df.columns if col.isdigit()]
            df = df.drop(columns=component_cols)

        bounds = await self._battery_soc_asset_bounds(mg_id, component_ids)
        if bounds is not None:
            lower_bound, upper_bound = bounds
            if lower_bound is not None:
                df["battery_soc_lower_bound_pct"] = lower_bound
            if upper_bound is not None:
                df["battery_soc_upper_bound_pct"] = upper_bound
        return df

    async def ac_active_energy(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        start: datetime,
        end: datetime,
        component_types: tuple[str, ...] = (
            "grid",
            "pv",
            "wind",
            "battery",
            "chp",
            "consumption",
        ),
        resampling_period: timedelta = timedelta(seconds=10),
        keep_components: bool = False,
        unit: str = "Wh",
        direction: Literal["net", "consumed", "delivered"] = "net",
    ) -> pd.DataFrame | None:
        """Fetch AC active energy, optionally for one direction only.

        direction="net" fetches the net cumulative-energy metric.  The raw
        readings are cumulative meter values.
        """
        if direction not in ("net", "consumed", "delivered"):
            raise ValueError(f"Unknown energy direction: {direction}")

        energy = await self.metric_data(
            microgrid_id=microgrid_id,
            start=start,
            end=end,
            component_types=component_types,
            resampling_period=resampling_period,
            metric=self._AC_ACTIVE_ENERGY_METRICS[f"energy_{direction}"],
            keep_components=keep_components,
        )
        if energy is None:
            return None
        energy = self._convert_units(
            energy, unit=unit, scale_by_unit={"Wh": 1, "kWh": 1000, "MWh": 1e6}
        )
        return energy

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
    async def _battery_metric_component_data(  # pylint: disable=too-many-arguments
        self,
        *,
        microgrid_id: int,
        component_ids: list[int],
        metric: Metric,
        metric_name: str,
        start: datetime,
        end: datetime,
        resampling_period: timedelta,
    ) -> pd.DataFrame | None:
        """Fetch a metric's values for battery components, indexed by timestamp."""
        data = [
            sample
            async for sample in self._client.receive_microgrid_components_data(
                microgrid_components=[(microgrid_id, component_ids)],
                metrics=metric,
                start_time=start,
                end_time=end,
                resampling_period=resampling_period,
            )
        ]
        if not data:
            _logger.warning(
                "No %s data found for microgrid %s.", metric_name, microgrid_id
            )
            return None

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates(keep="first").pivot_table(
            index="timestamp", columns="component_id", values="value"
        )
        for component_id in component_ids:
            if component_id not in df.columns:
                df.loc[:, component_id] = np.nan
        df = df.rename(columns=str)
        return df.reindex(columns=sorted(df.columns))

    @staticmethod
    def _capacity_weighted_soc(
        soc_df: pd.DataFrame, capacity_df: pd.DataFrame
    ) -> pd.Series | None:
        """Calculate aggregate battery SOC weighted by per-timestamp capacities."""
        component_cols = [col for col in capacity_df.columns if col in soc_df.columns]
        if not component_cols:
            _logger.warning(
                "Cannot calculate capacity-weighted battery SOC: no component "
                "SOC columns match battery capacity columns."
            )
            return None

        soc = soc_df[component_cols].apply(pd.to_numeric, errors="coerce")
        capacity = capacity_df[component_cols].apply(pd.to_numeric, errors="coerce")
        valid = soc.notna() & capacity.notna() & (capacity > 0)

        weighted_capacity = capacity.where(valid)
        available_capacity = weighted_capacity.sum(axis=1)
        weighted_sum = (soc.where(valid) * weighted_capacity).sum(axis=1, min_count=1)

        return cast(
            pd.Series,
            weighted_sum.div(available_capacity).where(available_capacity > 0),
        )

    @staticmethod
    def _align_forward_filled(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
        """Reindex a DataFrame to `index`, forward-filling from its own history."""
        combined_index = df.index.union(index)
        return df.reindex(combined_index).ffill().reindex(index)

    async def _battery_soc_asset_bounds(
        self,
        microgrid_id: int | str,
        component_ids: list[int],
    ) -> tuple[float | None, float | None] | None:
        """Fetch the tight static SOC interval from battery asset metadata."""
        if self._assets_client is None:
            _logger.warning(
                "Cannot fetch battery SOC bounds for microgrid %s: no Assets API client.",
                microgrid_id,
            )
            return None

        batteries = await self._assets_client.list_microgrid_electrical_components(
            MicrogridId(int(microgrid_id)),
            categories=[ElectricalComponentCategory.BATTERY],
        )
        configured_ids = set(component_ids)
        asset_bounds = [
            battery.rated_bounds.get(AssetsMetric.BATTERY_SOC_PCT)
            for battery in batteries
            if int(battery.id) in configured_ids
        ]
        lower_bounds = [
            bound.lower
            for bound in asset_bounds
            if bound is not None and bound.lower is not None
        ]
        upper_bounds = [
            bound.upper
            for bound in asset_bounds
            if bound is not None and bound.upper is not None
        ]
        if not lower_bounds and not upper_bounds:
            _logger.warning(
                "Assets API returned no battery SOC rated bounds for microgrid %s.",
                microgrid_id,
            )
            return None

        lower = max(lower_bounds) if lower_bounds else None
        upper = min(upper_bounds) if upper_bounds else None
        if lower is not None and upper is not None and lower > upper:
            _logger.warning(
                "Battery SOC rated bounds do not overlap for microgrid %s.",
                microgrid_id,
            )
            return None
        return lower, upper
