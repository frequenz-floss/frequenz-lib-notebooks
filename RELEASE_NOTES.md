# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

<!-- Here goes notes on how to upgrade from previous versions, including deprecations and what they should be replaced with -->

- `MicrogridConfig`: Switch to schema-based loading of microgrid config files and updates to the config class:
   - Remove unused nested field `assets` and replace by its contents `pv`, `wind`, `battery`.
   - Make `meta` and `ctype` public fields.
   - Require `meta.microgrid_id` to be set.
- The minimum supported version of `matplotlib` is now `v3.9.2`.

## New Features

<!-- Here goes the main new features and examples or instructions on how to use them -->

- This release supports python 3.13.

## Bug Fixes

<!-- Here goes notable bug fixes that are worth a special mention or explanation -->
