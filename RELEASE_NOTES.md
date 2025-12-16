# Tooling Library for Notebooks Release Notes

## Summary

- Align tooling with the latest reporting client, gridpool, and repository config releases while tightening typing and plotting helpers across the solar reporting suites.

## Upgrading

- Point dependencies at the new `frequenz-client-reporting >= 0.20` line, `frequenz-gridpool >= 0.1.1`, and `frequenz-repo-config 0.13.8` so packaging matches the newest releases.
- Update the default solar metric to `Metric.AC_POWER_ACTIVE` alongside the v2 `Metric` import path.

## New Features

- N/A

## Bug Fixes

- Fix nox formatting/type warnings by casting timezone conversions, enforcing literal aggregation functions, and keeping area/line plotting parameters typed in the solar maintenance views.
- Calculate consumption in scenarios where consumption is missing from the data.
