# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

<!-- Here goes notes on how to upgrade from previous versions, including deprecations and what they should be replaced with -->

## New Features

- Added options to add microgrid location metadata to `MicrogridConfig` (longitude, latitude, altitude).
- Added support for `component_category` in the `component_type_ids` method of `MicrogridConfig`, allowing retrieval of IDs for specific categories (e.g., "meter", "inverter", and "component").
- Added assets config to `MicrogridConfig`, which can be used to track metadata for assets like pv, wind and battery.

## Bug Fixes

<!-- Here goes notable bug fixes that are worth a special mention or explanation -->
