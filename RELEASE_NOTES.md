# Tooling Library for Notebooks Release Notes

## Summary

- Fixed asset optimization battery and power-flow rendering for more accurate charge/discharge visuals.

## Upgrading


## New Features

- Refactored asset optimization Plotly code to reduce duplication in layout finalization and battery trace creation.

## Bug Fixes
- Fixed asset optimization power-flow charge/discharge fills to be anchored to the consumption baseline while keeping hover values on actual series.
- Fixed battery power chart fills across missing data by inserting zero boundaries at NaN edges to avoid visual bridging through gaps.
- Fixed battery charge rendering to align positive charge fill with available power bounds in the asset optimization Plotly chart.
