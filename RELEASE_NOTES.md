# Tooling Library for Notebooks Release Notes

## Summary

This release adds formula-aware microgrid configuration initialization for
reporting data workflows. When API credentials are available, microgrid
configurations are enriched with formulas from the Assets service; otherwise
the existing static config loading path is used.

Also fixed asset optimization battery and power-flow rendering for more accurate charge/discharge visuals.

## Upgrading

- `init_microgrid_data()`
  - now conditionally loads configs with formulas when both `API_KEY` and `API_SECRET` are set.
  - now supports a file argument in addition to the folder argument.

## New Features

- Refactored asset optimization Plotly code to reduce duplication in layout finalization and battery trace creation.

## Bug Fixes
- Fixed asset optimization power-flow charge/discharge fills to be anchored to the consumption baseline while keeping hover values on actual series.
- Fixed battery power chart fills across missing data by inserting zero boundaries at NaN edges to avoid visual bridging through gaps.
- Fixed battery charge rendering to align positive charge fill with available power bounds in the asset optimization Plotly chart.
