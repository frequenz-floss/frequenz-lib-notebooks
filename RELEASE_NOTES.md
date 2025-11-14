# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

<!-- Here goes notes on how to upgrade from previous versions, including deprecations and what they should be replaced with -->

- `MicrogridConfig`: Switch to schema-based loading of microgrid config files and updates to the config class:
   - Remove unused nested field `assets` and replace by its contents `pv`, `wind`, `battery`.
   - Make `meta` and `ctype` public fields.
   - Require `meta.microgrid_id` to be set.
- The minimum supported version of `matplotlib` is now `v3.9.2`.
- Add `src/frequenz/lib/notebooks/reporting/schema_mapping.yaml` to your deployment so notebooks can load the canonical column definitions via `ColumnMapper`.

## New Features

<!-- Here goes the main new features and examples or instructions on how to use them -->

- Introduced `frequenz.lib.notebooks.reporting.metrics.reporting_metrics` with first-class definitions for production excess, battery charging share, grid feed-in, self-consumption, self-consumption share, and inferred consumption; the schema now documents each metric via an `implementation` tag.
- Added `frequenz.lib.notebooks.reporting.utils.helpers.add_energy_flows()` (plus supporting helpers) that aggregates raw production/consumption columns and appends the derived flows the reporting notebook needs.
- Published a locale-aware `ColumnMapper` utility that reads the YAML schema so notebooks can seamlessly move between raw API headers, canonical identifiers, and localized display labels.
- plot_time_series() can auto-pivot long-format inputs, honor a desired legend/trace order, and optionally fill selected traces while keeping Plotly colors consistent through a shared palette builder; also defaults to numeric columns only to avoid spurious traces (src/frequenz/lib/notebooks/reporting/plotter.py (lines 15-163)).
- Added reusable long_to_wide() and build_color_map() helpers so notebooks can pivot categorical telemetry and reuse the canonical color scheme without duplicating logic (src/frequenz/lib/notebooks/reporting/utils/helpers.py (lines 216-311)).

## Bug Fixes
- `frequenz.lib.notebooks.reporting.utils.helpers.add_energy_flows()` now infers consumption totals from existing data when explicit consumption columns are missing, preventing inconsistent outputs in notebook pipelines that only provide grid and production inputs.
- `frequenz.lib.notebooks.reporting.metrics.consumption()` reindexes optional production/battery inputs and raises a warning when inferred consumption turns negative so sign-convention issues are surfaced immediately.
- plot_energy_pie_chart() now accepts the same color_dict overrides as the time-series view, ensuring doughnut slices reuse the canonical colors instead of Plotly’s defaults and keeping legends consistent across charts (src/frequenz/lib/notebooks/reporting/plotter.py (lines 166-200)).

<!-- Here goes notable bug fixes that are worth a special mention or explanation -->
