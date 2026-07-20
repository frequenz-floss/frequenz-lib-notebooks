# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Typed metadata for component-oriented reporting outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ComponentMetadata:
    """Dataset-level metadata for component display and grouping."""

    display_names: dict[str, str] = field(default_factory=dict)
    ids_by_type: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class EnergyReport:
    """Normalized energy report dataframe plus resolved component metadata."""

    df: pd.DataFrame
    metadata: ComponentMetadata = field(default_factory=ComponentMetadata)
