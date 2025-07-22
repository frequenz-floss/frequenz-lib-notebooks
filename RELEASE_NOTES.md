# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

<!-- Here goes notes on how to upgrade from previous versions, including deprecations and what they should be replaced with -->

## New Features

- Added consistent logger setup across all modules for structured logging and improved observability. Example notebooks updated to demonstrate logger usage.
- The signature for passing config files MicrogridConfig.load_config() has been changed to accept a path a list of paths and a directory containing the config files.
- `MicrogridData` class needs to be initialized with a `MicrogridConfig` object instead of a path to config file(s).
- Added a transactional stateful data fetcher.
- Added a new `state_analysis` module for detecting and analysing component state transitions and alerts from reporting data.
  - Provides structured `StateRecord` objects with human-readable enum names.
  - Supports filtering for alert states and warnings.
  - Includes full test coverage for transition detection and alert filtering logic.

## Bug Fixes

- Fixed a bug in the notification `Scheduler` where tasks could overrun the configured duration due to imprecise sleep and stop logic. The scheduler now correctly tracks elapsed time, respects task execution duration, and stops reliably after the intended interval.
- Fixed an issue where `EmailNotification` did not properly initialise its scheduler. Also fixed an example in the docstring.
