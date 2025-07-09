# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

<!-- Here goes notes on how to upgrade from previous versions, including deprecations and what they should be replaced with -->

## New Features

<!-- Here goes the main new features and examples or instructions on how to use them -->

## Bug Fixes

- Fixed a bug in the notification `Scheduler` where tasks could overrun the configured duration due to imprecise sleep and stop logic. The scheduler now correctly tracks elapsed time, respects task execution duration, and stops reliably after the intended interval.
- Fixed an issue where `EmailNotification` did not properly initialise its scheduler. Also fixed an example in the docstring.