# Tooling Library for Notebooks

[![Build Status](https://github.com/frequenz-floss/frequenz-lib-notebooks/actions/workflows/ci.yaml/badge.svg)](https://github.com/frequenz-floss/frequenz-lib-notebooks/actions/workflows/ci.yaml)
[![PyPI Package](https://img.shields.io/pypi/v/frequenz-lib-notebooks)](https://pypi.org/project/frequenz-lib-notebooks/)
[![Docs](https://img.shields.io/badge/docs-latest-informational)](https://frequenz-floss.github.io/frequenz-lib-notebooks/)

## Introduction

A modular Python toolkit designed to support notebook-based workflows. It provides reusable tools for data ingestion, transformations, visualisation, notifications, and microgrid metadata managers. These tools make the repository ideal for streamlining analytics workflows with minimal setup and building data pipelines, reporting workflows, and alert systems seamlessly in Jupyter or cloud notebooks.

## Supported Platforms

The following platforms are officially supported (tested):

- **Python:** 3.11
- **Operating System:** Ubuntu Linux 20.04
- **Architectures:** amd64, arm64

## Contributing

If you want to know how to build this project and contribute to it, please
check out the [Contributing Guide](CONTRIBUTING.md).

## Quick Start

Install the package, open the example notebooks, and explore the available modules.

### Installation

```
# Choose the version you want to install
VERSION=0.9.2
pip install frequenz‑lib‑notebooks==$VERSION
```

Then open the prebuilt example notebooks using your preferred interface:
- Classic Notebook: `jupyter examples/`
- JupyterLab: `jupyter-lab examples/`

⚠️ **Note**: This project does **not** install `jupyter` or `jupyterlab` by default. You will need to install it separately if you want to run notebooks:

```
pip install jupyterlab
```

### Code Examples

#### 📧 Example 1: Generate an Alert Email (HTML Body Only)

This example shows how to:
- Transform a `pandas` DataFrame of alert records into a structured HTML email using `generate_alert_email`.
- Use `AlertEmailConfig` to control layout (e.g., table row limits, sorting by severity).
- Integrate microgrid-component alerts cleanly into operational workflows (e.g., for notifications or reporting tools).

```
import pandas as pd
from IPython.display import HTML

from frequenz.lib.notebooks.alerts.alert_email import (
    AlertEmailConfig,
    generate_alert_email,
)
from frequenz.lib.notebooks.notification_utils import format_email_preview

# Example alert records dataframe
alert_records = pd.DataFrame(
    [
        {
            "microgrid_id": 1,
            "component_id": 1,
            "state_type": "error",
            "state_value": "UNDERVOLTAGE",
            "start_time": "2025-03-14 15:06:30",
            "end_time": "2025-03-14 17:00:00",
        },
        {
            "microgrid_id": 2,
            "component_id": 1,
            "state_type": "state",
            "state_value": "DISCHARGING",
            "start_time": "2025-03-14 15:06:30",
            "end_time": None,
        },
    ]
)

# Configuration for email generation
alert_email_config = AlertEmailConfig(
    displayed_rows=10,
    sort_by_severity=True,
)

# Generate the HTML body of the alert email
html_email = generate_alert_email(
    alert_records=alert_records, config=alert_email_config
)

# Output the HTML # or send it via email as shown in the next example
print(html_email)

# or preview it in a nicer format
HTML(format_email_preview(subject="Alert Notification", body_html=html_email))
```

#### 📨 Example 2: Compose and Send Alert Email with Attachments
Continuing from Example 1, this snippet builds on the generated HTML email and demonstrates:

- Configuring SMTP credentials and recipients.
- Attaching both a CSV export of the alert data and optional visual plots.
- Sending the email once or scheduling it periodically. Note that the periodic scheduling would make sense when the data also refreshes so as to not send the same email over and over again!

```
import time
from datetime import datetime

from frequenz.lib.notebooks.alerts.alert_email import ExportOptions, plot_alerts
from frequenz.lib.notebooks.notification_service import (
    EmailConfig,
    EmailNotification,
    SchedulerConfig,
)

# Configuration for email notification
email_config = EmailConfig(
    subject="Critical Alert",
    message=html_email,  # Assuming that html_email already exists. See the code example above on how to generate this.
    recipients=["recipient@example.com"],
    smtp_server="smtp.example.com",
    smtp_port=587,
    smtp_user="user@example.com",
    smtp_password="password",
    from_email="alert@example.com",
    scheduler=SchedulerConfig(
        send_immediately=True,
        interval=60,  # send every minute
        duration=3600,  # for one hour total
    ),
)
# The SMTP details and sender/recipient details need to be adjusted accordingly
# note that the library provides a convenient way to validate the settings via frequenz.lib.notebooks.notification_utils.validate_email_config

# Create a notification object
email_notification = EmailNotification(config=email_config)

# optionally add attachments (a list of files)
email_config.attachments = None
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
alert_file_name = f"alert_details_{timestamp}.csv"
alert_records.to_csv(alert_file_name, index=False)
email_config.attachments = [alert_file_name]

# Optionally create a visualisation of the alert records
img_path = plot_alerts(
    records=alert_records,
    plot_type="all",
    export_options=ExportOptions(
        format="png",
        show=True,
    ),
)
email_config.attachments += img_path if img_path else []

# Send one-off notification
email_notification.send()

# Or start a periodic scheduler:
email_notification.start_scheduler()
time.sleep(300)  # let it run for 5 minutes
email_notification.stop_scheduler()
```

##  Module Overview

- **Solar Maintenance App:** Interactive forecasting and visualisation tools tailored to solar installations.
- **Notification Service:** Flexible and configurable email dispatching.
- **Alert Email Generation:** Embed rich Plotly charts into alert emails, complete with context and summaries.
- **Microgrid Configuration:** Manage structured microgrid metadata—location, devices, etc. — consistently across notebooks.

For more details about each module/project, please refer to the overview `Wiki` [page](https://github.com/frequenz-floss/frequenz-lib-notebooks/wiki/Frequenz-Lib-Notebooks-%E2%80%90-Overview) which has links to dedicated project pages.

The full code documentation can be accessed [here](https://frequenz-floss.github.io/frequenz-lib-notebooks/latest/).