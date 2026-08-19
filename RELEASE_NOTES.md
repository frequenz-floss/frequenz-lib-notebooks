# Tooling Library for Notebooks Release Notes

## Summary

- The solar maintenance workflow now renders its plots with Plotly instead of Matplotlib, providing interactive plots.
- The combined battery SOC and usecase reporting plot now uses Plotly Resampler to keep large time-series plots responsive while still loading detailed data when users zoom in.
- Reporting and Assets API credentials are now resolved from `FREQUENZ_API_KEY` and
  `FREQUENZ_API_SECRET`, with API-specific override pairs for individual services.

## Upgrading

- Solar maintenance plots are now Plotly figures. Code that directly accessed Matplotlib figure, axes, legend, colormap, or PNG-specific APIs from the solar maintenance plotting internals may need to be updated to use the Plotly-based wrappers or Plotly figure APIs.
- Plotly Resampler is now a runtime dependency. `plot_time_series_battery_soc_and_usecase()` returns a resampler-backed Plotly figure by default; pass `enable_resampler=False` to keep the previous plain `go.Figure` behavior.
- Replace legacy `API_KEY`/`API_SECRET` or `API_AUTH_KEY`/`API_SIGN_SECRET` variables with `FREQUENZ_API_KEY`/`FREQUENZ_API_SECRET` for the reporting, asset optimization, and solar maintenance workflows. Use `REPORTING_API_KEY`/`REPORTING_API_SECRET` or `ASSETS_API_KEY`/`ASSETS_API_SECRET` only when a service needs credentials that differ from the generic Frequenz API credentials.
- `MicrogridData.metric_data()` accepts `data_fetch_mode="energy"` to derive active power from consumed and delivered cumulative energy metrics.

## New Features

- Added interactive Plotly plots to the solar maintenance workflow, including per-subplot legends, unified hover boxes, compact axis tick labels, and full date values in hover labels.
- Added dynamic resampling to `plot_time_series_battery_soc_and_usecase()` so the initial figure payload is downsampled and zoom interactions resample from the high-frequency data without adding aggregation-size suffixes to legend labels.
- Added shared credential resolution for Frequenz APIs. Generic credentials are used by default, while a complete API-specific credential pair overrides them for that service.
- Added `MicrogridData.ac_active_energy_consumed()`, `MicrogridData.ac_active_energy_delivered()`, and `MicrogridData.ac_active_energy_net()` for active energy data.


## Bug Fixes
