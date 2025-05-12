# Tooling Library for Notebooks Release Notes

## Summary

This release refactors the solar maintenance workflow to return structured data via a new `SolarAnalysisData` dataclass and aggregates production statistics across multiple microgrids into a single table. The microgrid overview interface was updated to support multiple grids using a modular layout. Several fixes improve robustness against missing data and correct inconsistencies in production totals involving current-day values.

## Upgrading
- Refactored the solar maintenance workflow to aggregate production statistics from all requested microgrids into a single table instead of creating one table per microgrid.

## New Features

- Introduced the `SolarAnalysisData` dataclass to structure the output of the `solar_maintenance_app.run_workflow()` function.
- Added a modular `MicrogridOverviewDashboard` for dynamic multi-microgrid production display with light/dark theme support. Replaces the current hardcoded single-microgrid layout.

## Bug Fixes

- Introduced `NoDataAvailableError` exception to represent situations where no data is available and to skip such cases during workflow execution and plotting.
- Fixed a bug introduced in version `v0.5.2` where the microgrid tabular overview excluded current-day values from the total production for the past 30 and 365 days, causing inconsistencies with metrics that included today's data.
- Fixed a crash in statistical plot style generation when the input data index is empty or contains only a single timestamp. Duration is now set to `None` when no `mode` can be determined.
