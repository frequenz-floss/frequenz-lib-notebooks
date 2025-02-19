# Tooling Library for Notebooks Release Notes

## Summary

Renamed the `"load"` component type to `"consumption"` for clarity.

## Upgrading

- If your code references `"load"` as a component type, update it to `"consumption"`.
- Made the `MicrogridConfig` reader tolerant to missing `ctype` fields, allowing collection of incomplete microgrid configs.

## New Features

<!-- Here goes the main new features and examples or instructions on how to use them -->

## Bug Fixes

<!-- Here goes notable bug fixes that are worth a special mention or explanation -->
