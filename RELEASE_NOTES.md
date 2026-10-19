# Tooling Library for Notebooks Release Notes

## Summary


## Upgrading

- Energy reports now default to `grid_data_source="energy"`. Notebooks must
  fetch their data with `fetch_energy_report_data(...)` instead of
  `mdata.ac_active_power(...)`, so the frame contains the
  `grid_energy_consumed` and `grid_energy_delivered` columns. Pass the same
  `resampling_period` as `resolution` to `create_energy_report_df(...)`
  (both default to 10 seconds; a mismatch silently mis-scales the grid power).
  To keep the previous behaviour, pass `grid_data_source="power"`.
- `add_energy_flows(...)` no longer accepts the `grid_cols` parameter. Grid columns are now selected through `grid_data_source`.
- `battery_power_flows(...)` now takes explicit grid import and export series,
  and `grid_to_battery` is capped by grid import. Battery-to-consumption and
  grid-to-battery results may differ from previous releases.

## New Features
- Energy reports now use directional active-energy readings for grid data by
  default, while all other component data remains active power. Set
  `grid_data_source="power"` to use signed grid power instead.


## Bug Fixes
- Separate out positive values as grid consumption from the common grid column.
