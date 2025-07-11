# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

<!-- Here goes notes on how to upgrade from previous versions, including deprecations and what they should be replaced with -->

## New Features

- Added consistent logger setup across all modules for structured logging and improved observability. Example notebooks updated to demonstrate logger usage.

## Bug Fixes

- Fixed a bug in the notification `Scheduler` where tasks could overrun the configured duration due to imprecise sleep and stop logic. The scheduler now correctly tracks elapsed time, respects task execution duration, and stops reliably after the intended interval.
- Fixed an issue where `EmailNotification` did not properly initialise its scheduler. Also fixed an example in the docstring.