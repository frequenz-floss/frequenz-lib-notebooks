# Tooling Library for Notebooks Release Notes

## Summary

This release syncs the reporting helpers with the latest schema and dependency requirements so that asset columns, component analyses, and timestamp handling all stay consistent with the data sources we consume.

## Upgrading

- The pyproject lock now requires `pyyaml>=6.0.3` and `pytz>=2025.2`, so reinstall the dependencies to pick up the tighter YAML and timezone support.
- Column names in exported energy reports have shifted: `grid_consumption` is now computed via the `grid` helper column before renaming, and battery data should be referenced as `battery_power_flow` rather than `battery_throughput`.

## New Features

- Wind asset production is now fleshed out in the schema mapping (including localized display names) and is available to the overview builder and component analysis flows, alongside the existing PV and CHP columns.
- Component analyses now apply the mapper renaming step after totals are summed, preventing columns from being renamed prematurely and ensuring the display names defined in `schema_mapping.yaml` are always respected.

## Bug Fixes

- Notification deserialization now relies on `isinstance()` to guard against optional dataclass arguments, eliminating the brittle `type()` checks that occasionally crashed the signal service.
- Peak-date computation now pulls the timestamp directly from the aggregated dataframe, coerces it to UTC, and formats it safely, so peak dates no longer break when indexes are strings or timezone-naive.
- The grid column is only populated when it is missing to avoid re-creating `grid_consumption`, preventing duplicate columns during the energy-flow aggregation.
- Schema mappings for CHP and wind outputs have been aligned with the latest raw column tags, and all production display names now use the new hyphenated nomenclature (`PV-Production`, `CHP-Production`, `Wind-Production`).
