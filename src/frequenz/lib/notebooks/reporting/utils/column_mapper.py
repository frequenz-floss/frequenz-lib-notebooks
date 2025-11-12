# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Column mapping utilities for energy reporting.

Keeps reporting notebooks insulated from upstream schema churn
by loading the canonical schema from
`src/frequenz/lib/notebooks/reporting/schema_mapping.yaml`. The class translates
raw headers to canonical names, resolves locale-aware display labels.
Raw names are the column labels emitted directly by the reporting API, canonical
names are the identifiers used throughout this codebase, and the display names
(`en`/`de`) expose English and German labels rendered in the notebooks. The
mapper is designed to work hand-in-hand with the schema_mapping.yaml file so that
every canonical column carries consistent units and descriptions.

Examples:
    >>> from frequenz.lib.notebooks.reporting.utils import ColumnMapper
    >>> mapper = ColumnMapper.from_yaml(
    ...     "src/frequenz/lib/notebooks/reporting/schema_mapping.yaml",
    ...     locale="de",
    ... )
    >>> display_df = mapper.to_display(mapper.to_canonical(raw_df))
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib import resources
from typing import Iterable, Mapping

import pandas as pd
import yaml


@dataclass(frozen=True)
class ColumnMapper:  # pylint: disable=too-many-instance-attributes
    """Column schema with locale-aware display labels."""

    version: int = field(
        metadata={"doc": "Schema version extracted from the YAML definition."}
    )
    tz_name: str = field(
        metadata={
            "doc": "Timezone name to apply when localizing timestamps in reporting notebooks.",
        }
    )
    assume_tz: str = field(
        metadata={
            "doc": "Timezone assumed for raw timestamps before conversion to tz_name.",
        }
    )
    canonical_to_raw: Mapping[str, str] = field(
        metadata={
            "doc": (
                "Mapping from canonical column identifiers to raw headers "
                "from the reporting API."
            ),
        }
    )
    raw_to_canonical: Mapping[str, str] = field(
        metadata={
            "doc": "Reverse lookup mapping raw API headers back to canonical identifiers.",
        }
    )
    _labels_all: Mapping[str, Mapping[str, str]] = field(
        metadata={
            "doc": "Locale-specific display labels keyed by canonical column name.",
        }
    )
    locale: str = field(
        default="de",
        metadata={"doc": "Preferred locale for display labels when renaming columns."},
    )
    fallback_locale: str = field(
        default="en",
        metadata={
            "doc": "Fallback locale to use when the preferred locale is unavailable in the schema.",
        },
    )
    _DEFAULT_RESOURCE = resources.files("frequenz.lib.notebooks.reporting").joinpath(
        "schema_mapping.yaml"
    )

    # ---------- Construction ----------
    @classmethod
    def from_yaml(  # pylint: disable=too-many-locals
        cls,
        path: str,
        *,
        locale: str = "de",
        fallback_locale: str = "en",
        required: Iterable[str] | None = None,
    ) -> "ColumnMapper":
        """
        Create a ColumnMapper from a YAML configuration file.

        Args:
            path: Path to the YAML file containing the column mapping definition.
            locale: Preferred display locale (default: "de").
            fallback_locale: Fallback locale if the preferred one is missing
                (default: "en").
            required: Optional list of required canonical column names. Raises
                ValueError if any are missing.

        Returns:
            A ColumnMapper instance built from the YAML configuration.

        Raises:
            ValueError: If the YAML is missing required fields or contains invalid mappings.
        """
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cols = cfg.get("columns") or {}
        if not isinstance(cols, dict) or not cols:
            raise ValueError("YAML 'columns' is missing or empty.")

        c2r: dict[str, str] = {}
        labels_all: dict[str, dict[str, str]] = {}

        for canonical, spec in cols.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid spec for '{canonical}' (must be mapping).")
            raw = spec.get("raw")
            if not raw:
                raise ValueError(f"'raw' missing for canonical column '{canonical}'.")
            c2r[canonical] = raw

            disp = spec.get("display") or {}
            if not isinstance(disp, dict):
                raise ValueError(f"'display' for '{canonical}' must be a mapping.")
            labels_all[canonical] = {str(k): str(v) for k, v in disp.items()}

        # Build reverse map and check collisions
        r2c: dict[str, str] = {}
        for c, r in c2r.items():
            if r in r2c and r2c[r] != c:
                raise ValueError(f"Raw column '{r}' maps to both '{r2c[r]}' and '{c}'.")
            r2c[r] = c

        if required:
            missing = set(required) - set(c2r.keys())
            if missing:
                raise ValueError(
                    f"Missing required canonical columns: {sorted(missing)}"
                )

        time_cfg = cfg.get("time") or {}
        return cls(
            version=int(cfg.get("version", 0)),
            tz_name=str(time_cfg.get("tz_name", "UTC")),
            assume_tz=str(time_cfg.get("assume_tz", "UTC")),
            canonical_to_raw=c2r,
            raw_to_canonical=r2c,
            _labels_all=labels_all,
            locale=locale,
            fallback_locale=fallback_locale,
        )

    @classmethod
    def from_default(
        cls,
        *,
        locale: str = "de",
        fallback_locale: str = "en",
        required: Iterable[str] | None = None,
    ) -> ColumnMapper:
        """Create an instance using the built-in default YAML resource.

        Loads the default configuration file bundled with the package and
        initializes the class using localized settings.

        Args:
            locale: Primary locale code for localized configuration values.
                Defaults to "de".
            fallback_locale: Secondary locale used if a key is missing in
                the primary locale. Defaults to "en".
            required: Optional list of keys that must be present in the
                loaded configuration. If provided and any are missing,
                an exception is raised.

        Returns:
            An initialized instance of the class populated from the
            built-in default YAML resource.
        """
        with resources.as_file(cls._DEFAULT_RESOURCE) as yaml_path:
            return cls.from_yaml(
                str(yaml_path),
                locale=locale,
                fallback_locale=fallback_locale,
                required=required,
            )

    # ---------- Properties ----------
    @property
    def canonical_to_display(self) -> Mapping[str, str]:
        """Resolved display labels for the current locale (with fallback)."""
        return {
            c: (
                self._labels_all.get(c, {}).get(self.locale)
                or self._labels_all.get(c, {}).get(self.fallback_locale)
                or c
            )
            for c in self.canonical_to_raw
        }

    @property
    def canonicals(self) -> Iterable[str]:
        """Return the canonical column names defined in this mapper."""
        return self.canonical_to_raw.keys()

    # ---------- Operations on DataFrames ----------
    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename incoming raw headers to canonical headers (normalize once)."""
        return df.rename(columns=self.raw_to_canonical)

    def to_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename canonical headers back to raw."""
        return df.rename(columns=self.canonical_to_raw)

    def to_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename canonical headers to localized display labels."""
        return df.rename(columns=self.canonical_to_display)

    # ---------- Locale switching ----------
    def with_locale(
        self, locale: str, fallback_locale: str | None = None
    ) -> "ColumnMapper":
        """Create a copy with a different display locale (no re-read of YAML)."""
        return replace(
            self,
            locale=locale,
            fallback_locale=(fallback_locale or self.fallback_locale),
        )
