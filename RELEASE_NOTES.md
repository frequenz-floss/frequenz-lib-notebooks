# Tooling Library for Notebooks Release Notes

## Summary


## Upgrading

- `init_microgrid_data` reads credentials from `API_AUTH_KEY`/`API_SIGN_SECRET`, falling back to `API_KEY`/`API_SECRET`. Both halves must come from the same pair.
- Dotenv files passed to `init_microgrid_data` now override the environment and each other, so pass the most specific file last.
- `plot_time_series_battery_usecase` now defaults to `stack_mode="psc"`.
  Pass `stack_mode="energy_balance"` to get an upward supply-balance view.

## New Features

- Adding wind data to data fetching.
- Add day ahead prices fetching.
- `aggregate_metrics` now exposes grid import cost and grid feed-in revenue
  totals when price data is present in the reporting dataframe.
- Reporting notebooks can now build component-level analysis for configured
  meters, inverters, and other component groups selected from the microgrid
  configuration.
- Added `build_energy_report`, `ComponentMetadata`, and `EnergyReport` to
  expose canonical component IDs and optional meter display names from the
  Assets API to downstream reporting workflows.
- `create_energy_report_df` and downstream component analysis now preserve
  canonical component IDs, making notebook selections and plots easier to read.
- Add `plot_power` to the asset optimization plotly visualizations, stacking each component in the passive sign convention so the top of the stack meets the grid line. Unlike `plot_power_flow`, production is not clipped.
- `init_microgrid_data` accepts several dotenv files, so shared API URLs can live apart from per-microgrid credentials.
- `plot_time_series_battery_usecase` now supports selectable stack modes for
  battery overlays, including the new passive-sign-convention default and the
  legacy energy-balance mode.

## Bug Fixes

- `init_microgrid_data` no longer reuses the credentials of a previously loaded dotenv file, which made switching microgrid in a running notebook kernel keep the first microgrid's key.
- `~` in a dotenv path is expanded instead of silently loading nothing.
- Battery use-case dataframe generation now keeps charge/discharge series aligned
  with the plotting sign convention.
