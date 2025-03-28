# Tooling Library for Notebooks Release Notes

## Summary

The `frequenz-api-weather` package has been removed in favor of `frequenz-client-weather` because the required methods have been moved over to the latter library. Added `notification_utils.py` with validation, preview, and test email helpers. Updated "Alert Notebook.ipynb" in the `examples/` to utilise the new features.

## Upgrading

- The minimum required version of `frequenz-client-reporting` is now `v0.14.0`.

## New Features

* Add an example notebook with a simple PV forecast using weather API.
* Added `notification_utils.py` with:
    - `send_test_email()` for verifying SMTP configuration.
    - `validate_email_config()` for runtime validation of required fields, attachment presence, and optional SMTP connectivity.
    - `format_email_preview()` for generating simple HTML previews.
* Enhanced alert email generation with helpers for status and no-alert messages, and a new `AlertEmailConfig` dataclass for modularity and a cleaner interface.
* Added alert visualisation support via `alert_email.plot_alerts()`, enabling interactive summary and state transition plots (using `plotly`) with optional image export to various supported formats.

## Bug Fixes

<!-- Here goes notable bug fixes that are worth a special mention or explanation -->
