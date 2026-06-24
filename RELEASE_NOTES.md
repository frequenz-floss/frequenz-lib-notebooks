# Tooling Library for Notebooks Release Notes

## Summary

This release adds formula-aware microgrid configuration initialization for
reporting data workflows. When API credentials are available, microgrid
configurations are enriched with formulas from the Assets service; otherwise
the existing static config loading path is used.

It also adds a dedicated reporting battery-usecase plot, including standardized
data preparation and support for PV and battery overlays.

## Upgrading

- `init_microgrid_data()`
  - now conditionally loads configs with formulas when both `API_KEY` and `API_SECRET` are set.
  - now supports a file argument in addition to the folder argument.

## New Features

- Refactored asset optimization Plotly code to reduce duplication in layout finalization and battery trace creation.
- `plot_time_series_battery_usecase()`
  - added as a dedicated reporting plot for battery-usecase analysis.
- `create_battery_usecase_df()`
  - added to prepare standardized input data for the battery-usecase plot.

## Bug Fixes
- Fixed asset optimization power-flow charge/discharge fills to be anchored to the consumption baseline while keeping hover values on actual series.
- Fixed battery power chart fills across missing data by inserting zero boundaries at NaN edges to avoid visual bridging through gaps.
- Fixed battery charge rendering to align positive charge fill with available power bounds in the asset optimization Plotly chart.
- Fixed timezone-related datetime usage across notification and solar maintenance helpers by using explicit UTC-aware datetimes for defaults and generated timestamps.
