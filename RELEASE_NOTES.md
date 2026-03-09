# Tooling Library for Notebooks Release Notes

## Summary

- Refined asset optimization Plotly visuals and layout behavior for clarity and consistency.
- Updated reporting time-series visuals (colors, line styles, stacking, and shading).

## Upgrading

- Reporting metrics now consistently assume PSC sign conventions for production and no longer accept a `production_is_positive` toggle.
- Removed `production_is_positive` from reporting metric helpers and `add_energy_flows()`.
- If you previously passed already-positive production to these helpers, pass `-production` instead (PSC convention: production is negative).
- `plot_time_series()` now supports `shade_by_category` for long-format plots to render category series as shades of one base color.

## New Features

- Added production self-usage metrics to reporting flows and aggregations (`production_self_usage`, `production_self_share`).
- Improved asset optimization Plotly styling (fonts, hover labels, legends, titles) and subplot spacing.
- Added stacked PV/CHP behavior in asset optimization power flow plots (PV now stacks on CHP).
- Added `LINE_DASH_MAP` for consistent line dash styles and applied it in reporting time-series plots.
- Added per-category shade rendering for long-format reporting time-series plots.

## Bug Fixes
- Fixed monthly aggregation Plotly plot to group bars by month correctly.
- Fixed reporting time-series shading to correctly parse `rgb(...)`/`rgba(...)` base colors.
- Set reporting and asset optimization Plotly trace line widths to `1` for consistent stroke weight.
- Updated reporting color mappings to the latest conventions (incl. battery charge/discharge and PV/CHP/Wind labels).
