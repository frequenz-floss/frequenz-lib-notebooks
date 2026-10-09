# Tooling Library for Notebooks Release Notes

## Summary


## Upgrading
- When constructing `MicrogridData` directly, pass an `AssetsApiClient` through
  the optional `assets_client` argument to include SOC bounds. Aggregate SOC is
  calculated from configured components' `BATTERY_CAPACITY` values in Reporting.
- A `BATTERY_SOC_PCT` formula is no longer used for SOC reporting and can be
  removed. SOC values are now fetched from configured battery components.

Active-energy access is consolidated in `ac_active_energy()`. Use
`direction="net"` (the default), `"consumed"`, or `"delivered"` instead of the
separate active-energy fetchers.

## New Features
- Update reporting notebook with the latest changes.
- `ac_active_power()` can now derive power from net cumulative AC active energy
with `from_energy=True`. This supports a gradual migration of power-based
reporting; callers that need to avoid interpreting large energy changes as
instantaneous power spikes should consume `ac_active_energy()` data directly.
- Add lower and upper battery SOC rated bounds to energy reports and SOC plots.
- Calculate aggregate battery SOC as a capacity-weighted average of the SOC
  values fetched directly for configured battery components.

## Bug Fixes
