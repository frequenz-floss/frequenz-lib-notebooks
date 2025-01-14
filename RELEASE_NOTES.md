# Tooling Library for Notebooks Release Notes

## Summary

This is the first release of the library with the Solar Maintenance App! 🎉
The repo provides tools to monitor and maintain solar energy systems with the following key features:
- **Data Fetching and Processing:** Retrieve and preprocess weather and solar production data from external APIs (`frequenz-api-weather` and `frequenz-client-reporting`).
- **Prediction Model Preparation:** Prepare basic time series models for solar power predictions.
- **Visualisation Tools:** Generate calendar views, rolling averages, production profiles and daily statistics available via the `*Plotter` classes. Control what/how much data to show and how to display it using the corresponding configuration options. Customise the plots using tools like `PlotStyleStrategy` and `PlotManager`.
- **Translation Support:** Enable English and German translations via the `TranslationManager` class, for controlling all text displayed on the plots and tables.
- **Single Entry Point:** Integrate data fetching, processing and visualisations into a main workflow.
- **Notification Service:** Send alert notifications via email with support for scheduling and retries, including a linear backoff mechanism.

This release provides tools to solar system operators to monitor performance and track trends, and lays the groundwork for identifying potential system issues.

Moreover it includes a microgrid config module that contains component configs to get component IDs and formulas for component types (e.g. PV, battery) and metadata information (e.g. gridpool ID).
