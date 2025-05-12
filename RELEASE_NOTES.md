# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading
* Refactored the solar maintenance workflow to aggregate production statistics from all requested microgrids into a single table instead of creating one table per microgrid.

## New Features

- Introduced a `SolarAnalysisData` dataclass to structure the output of the `solar_maintenance_app.run_workflow()` function. This introduces a breaking change in the `Solar Maintenance.ipynb` notebook.
- Added a modular `MicrogridOverviewDashboard` for dynamic multi-microgrid production display with light/dark theme support. Replaces the current hardcoded single-microgrid layout.

## Bug Fixes

- Introduced `NoDataAvailableError` exception to represent situations where no data is available and to skip such cases during workflow execution and plotting.
- Fixed a bug introduced in version `v0.5.2` where the microgrid tabular overview excluded current-day values from the total production for the past 30 and 365 days, causing inconsistencies with metrics that included today's data.
