# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

<!-- Here goes notes on how to upgrade from previous versions, including deprecations and what they should be replaced with -->

## New Features

<!-- Here goes the main new features and examples or instructions on how to use them -->

Introduce a `ColumnMapper` for reporting notebooks that keeps raw column churn
out of notebook code. The mapper centralises canonical column names, locale-
specific display labels, and timezone metadata loaded from
`src/frequenz/lib/notebooks/reporting/schema_mapping.yaml`. It ships together
with metric definitions so dashboards use consistent labels and units when the
underlying data sources evolve.

## Bug Fixes

<!-- Here goes notable bug fixes that are worth a special mention or explanation -->
