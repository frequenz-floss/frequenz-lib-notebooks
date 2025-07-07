# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

- Updated the Alerts Notebook. You can now process multiple microgrid TOML files.

## New Features

- Added support for Slack notifications with threading and file attachments. Includes a new `SlackNotification` class and `SlackConfig` schema, supporting both webhook and token-based delivery. Includes example usage and test coverage. Introduced `slack_sdk` and `requests` as new dependencies.

## Bug Fixes

- Updated the Solar Maintenance notebook to fix the expected environment variable name for the reporting server url.
