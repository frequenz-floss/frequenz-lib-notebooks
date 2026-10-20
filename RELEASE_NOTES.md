# Tooling Library for Notebooks Release Notes

## Summary


## Upgrading
- When constructing `MicrogridData` directly, pass an `AssetsApiClient` through the optional `assets_client` argument to include SOC bounds. Aggregate SOC is calculated from configured components' `BATTERY_CAPACITY` values in Reporting.
- A `BATTERY_SOC_PCT` formula is no longer used for SOC reporting and can be removed. SOC values are now fetched from configured battery components.

Active-energy access is consolidated in `ac_active_energy()`. Use `direction="net"` (the default), `"consumed"`, or `"delivered"` instead of the separate active-energy fetchers. The method returns the energy used during each reporting period, calculated from the cumulative meter readings.

## New Features
- Update reporting notebook with the latest changes.
- `ac_active_power()` can now derive power from AC active energy deltas with `from_energy=True`.
- Energy reports can switch between signed grid power and directional energy with `grid_data_source="power"` or `grid_data_source="energy"`.
- Add lower and upper battery SOC rated bounds to energy reports and SOC plots.
- Calculate aggregate battery SOC as a capacity-weighted average of the SOC
  values fetched directly for configured battery components.

## Bug Fixes
- Update the color schema for battery charging and discharging plot.