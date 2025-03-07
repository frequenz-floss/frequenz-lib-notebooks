# Tooling Library for Notebooks Release Notes

## Summary


## Upgrading

* Made the `MicrogridConfig` reader tolerant to missing `ctype` fields, allowing collection of incomplete microgrid configs.
* Formula configs are now defined per metric to support different formulas for each metric in the same config file.
  This is a breaking change which requires updating the formula fields in the config file.
* Default formulas are defined for AC active power and battery SoC metrics.
  The default SoC calculation uses simple averages and ignore different battery capacities.
* The `cids` method is changed to support getting metric-specific CIDs which in this case are extracted from the formula.

## New Features

<!-- Here goes the main new features and examples or instructions on how to use them -->

## Bug Fixes

<!-- Here goes notable bug fixes that are worth a special mention or explanation -->
