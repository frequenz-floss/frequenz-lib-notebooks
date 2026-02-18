# Tooling Library for Notebooks Release Notes

## Summary

This release updates the asset-optimization reporting visuals to Plotly, improves
interactivity and styling, and refactors data preparation for reuse.

## Upgrading

- Asset-optimization plotting now returns Plotly figures (instead of matplotlib).
  Update any downstream code that expects matplotlib `Axes` objects.

## New Features

- Plotly-based asset-optimization charts with interactive hover, range sliders,
  and improved styling (white background, borders, legend layout).

## Bug Fixes
- Update the asset optimization notebook with the correct api keys.
