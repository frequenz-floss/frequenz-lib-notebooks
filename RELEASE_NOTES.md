# Tooling Library for Notebooks Release Notes

## Summary

## Upgrading

* Expose MicrogridData from microgrid namespace.
* Upgrade `frequenz-client-reporting` to minimum `v0.16.0`.
* Small updates to the Alerts Notebook and Solar Maintenance notebook.

## New Features
* Notification service: Introduced `NotificationSendError` for structured retry failure handling and updated the logging in `send_test_email()` utility function.
* Rolling and profile plots now include production data up to the latest available timestamp, even if the current day is incomplete. This provides more real-time views of today's production without waiting for the day to finish.

## Bug Fixes

* Fixed a bug where the Solar Maintenance app would crash if some requested inverter components were missing from the reporting data. Missing components are now handled gracefully with a warning.
* Fixed a bug in prediction model preparation where predictions could fail or mismatch the expected index when using minimal data and a large moving average window. Predictions are now correctly aligned with the data index, with missing values padded with NaN.
