# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

<!-- Here goes notes on how to upgrade from previous versions, including deprecations and what they should be replaced with -->

- The minimum supported version of `matplotlib` is now `v3.9.2`.
- Add `src/frequenz/lib/notebooks/reporting/schema_mapping.yaml` to your deployment so notebooks can load the canonical column definitions via `ColumnMapper`.

## New Features

<!-- Here goes the main new features and examples or instructions on how to use them -->

- Introduced `frequenz.lib.notebooks.reporting.metrics.reporting_metrics` with first-class definitions for production excess, battery charging share, grid feed-in, self-consumption, self-consumption share, and inferred consumption; the schema now documents each metric via an `implementation` tag.
- Added `frequenz.lib.notebooks.reporting.utils.helpers.add_energy_flows()` (plus supporting helpers) that aggregates raw production/consumption columns and appends the derived flows the reporting notebook needs.
- Published a locale-aware `ColumnMapper` utility that reads the YAML schema so notebooks can seamlessly move between raw API headers, canonical identifiers, and localized display labels.

## Bug Fixes

<!-- Here goes notable bug fixes that are worth a special mention or explanation -->
