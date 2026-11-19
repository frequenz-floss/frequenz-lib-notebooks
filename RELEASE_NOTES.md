# Tooling Library for Notebooks Release Notes

## Summary

Improve reporting and microgrid data-loading performance and resilience.

## Upgrading

- `add_energy_flows(...)` and `create_energy_report_df(...)` take new
  `grid_data_source` and `resolution` parameters. The default,
  `grid_data_source="power"`, keeps the previous behaviour, so existing
  notebooks keep working unchanged.
- `add_energy_flows(...)` no longer accepts the `grid_cols` parameter. Grid columns are now selected through `grid_data_source`.
- The grid metric helpers take directional series: `consumption(...)` needs
  `grid_import` and `grid_export`, `grid_consumption(...)` takes `grid_import`,
  and `grid_feed_in(...)` takes `grid_export` as its first argument.
- `MicrogridData.ac_active_energy(...)` now requires `direction`
  (`"consumed"` or `"delivered"`) and returns the energy used per period
  instead of cumulative readings. The `"net"` direction,
  `ac_active_power(from_energy=...)` and the `energy_net` metric are removed.
- `MicrogridData` methods require an integer `microgrid_id`; numeric strings are
  no longer accepted.
- `battery_power_flows(...)` now takes explicit grid import and export series,
  and `grid_to_battery` is capped by grid import. Battery-to-consumption and
  grid-to-battery results may differ from previous releases.

## New Features
- Fetch independent reporting and microgrid data queries concurrently to reduce data-loading time.
- Avoid duplicate per-component metric requests when a component is present in multiple formulas.
- Energy reports can use directional active-energy readings for grid data
  instead of signed grid power. Fetch the data with
  `MicrogridData.ac_active_power(..., grid_data_source="energy")` and pass the
  same `grid_data_source` and the fetch `resampling_period` as `resolution` to
  `create_energy_report_df(...)`. The final period, which has no energy delta,
  is dropped. A warning is logged when a grid energy counter
  decreases (e.g. a meter reset or dropout to zero); use
  `MicrogridData.ac_active_energy_interpolated` to repair such readings.

## Bug Fixes
