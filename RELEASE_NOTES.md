# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

- Updated the Alerts Notebook. You can now process multiple microgrid TOML files.

## New Features

<!-- Here goes the main new features and examples or instructions on how to use them -->

## Bug Fixes

- Updated the Solar Maintenance notebook to fix the expected environment variable name for the reporting server url.
- Fixed a bug in the notification `Scheduler` where tasks could overrun the configured duration due to imprecise sleep and stop logic. The scheduler now correctly tracks elapsed time, respects task execution duration, and stops reliably after the intended interval.
