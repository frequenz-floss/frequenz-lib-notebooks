# Tooling Library for Notebooks Release Notes

## Summary
This release standardizes energy reporting with a new suite of metric definitions, visualization helpers, and a schema-based ColumnMapper for consistent data handling. It also introduces breaking changes to MicrogridConfig to support schema-based loading and adds official support for Python 3.13.

## Upgrading
- `MicrogridConfig`: Switch to schema-based loading of microgrid config files and updates to the config class:
   - Remove unused nested field `assets` and replace by its contents `pv`, `wind`, `battery`.
   - Make `meta` and `ctype` public fields.
   - Require `meta.microgrid_id` to be set.
- The minimum supported version of `matplotlib` is now `v3.9.2`.
- Add `src/frequenz/lib/notebooks/reporting/schema_mapping.yaml` to your deployment so notebooks can load the canonical column definitions via `ColumnMapper`.

## New Features
- Introduced `frequenz.lib.notebooks.reporting.metrics.reporting_metrics` with first-class definitions for production excess, battery charging share, grid feed-in, self-consumption, self-consumption share, and inferred consumption; the schema now documents each metric via an `implementation` tag.
- Added `frequenz.lib.notebooks.reporting.utils.helpers.add_energy_flows()` (plus supporting helpers) that aggregates raw production/consumption columns and appends the derived flows the reporting notebook needs.
- Published a locale-aware `ColumnMapper` utility that reads the YAML schema so notebooks can seamlessly move between raw API headers, canonical identifiers, and localized display labels.
- plot_time_series() can auto-pivot long-format inputs, add a desired legend/trace order, and optionally fill selected traces while keeping Plotly colors consistent through a shared palette builder; also defaults to numeric columns only to avoid spurious traces.
- Added reusable `long_to_wide()` and `build_color_map()` helpers so notebooks can pivot categorical telemetry and reuse the canonical color scheme without duplicating logic.
- Added `create_energy_report_df()` to convert raw microgrid exports into timezone-aware, canonical energy-report tables with derived grid/battery KPIs and labeled component columns, letting dashboards bind to a consistent schema without bespoke glue.
- Introduced the `reporting_nb_functions` toolkit so notebooks can build overview tables, melt component selections, compute energy-mix summaries, and aggregate KPIs (production totals, self-consumption share, grid import peaks) for stakeholder-ready reporting pages.
- Expanded the reporting helper utilities with YAML config loading, German number formatting, timezone conversion, component labeling, energy-report column selection, and robust energy-flow derivations so multiple notebooks can reuse the same preprocessing primitives.
- Published `Reporting NB.ipynb` example that wires the mapper, helper utilities, and KPI builders together in a ready-to-run reporting notebook.

## Bug Fixes
- `plot_energy_pie_chart()` now accepts the same color_dict overrides as the time-series view, ensuring doughnut slices reuse the canonical colors instead of Plotly’s defaults and keeping legends consistent across charts.
