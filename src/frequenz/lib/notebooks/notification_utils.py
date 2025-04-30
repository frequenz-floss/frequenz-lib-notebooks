# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
Utility functions to support testing, validating, and previewing email notifications.

These are intended to be used in notebooks, Streamlit apps, or other tools that
help users configure and debug their notification settings interactively.
"""
import html
import logging
import os
import re
import smtplib
import traceback
from dataclasses import fields
from datetime import UTC, datetime

from frequenz.lib.notebooks.notification_service import (
    EmailConfig,
    EmailNotification,
    NotificationSendError,
)

_log = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def send_test_email(config: EmailConfig, verbose: bool = True) -> bool:
    """Send a test email using the given EmailConfig.

    The email message is generated automatically based on the provided
    configuration and replaces the provided one.

    Args:
        config: An EmailConfig instance.
        verbose: If True, prints the result to stdout.

    Returns:
        True if the email was sent successfully, False otherwise.
    """
    if config.subject is None:
        config.subject = "Test Email"

    config.message = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <h2 style="color: #007bff;">{config.subject}</h2>
        <p>
            This is a test email sent from the notification service to verify your email settings.
        </p>
        <p>
            <strong>
                If you received this email successfully, your SMTP configuration is correct.
            </strong>
        </p>

        <hr style="border: 1px solid #ddd;">

        <p><strong>✉️ Sent from:</strong> {config.from_email}</p>
        <p><strong>📩 Sent to:</strong> {config.recipients}</p>
        <p><strong>⏳ Timestamp:</strong> {datetime.now().astimezone(UTC)}</p>

        <hr style="border: 1px solid #ddd;">

        <p style="color: #666;">
            <em>If you did not initiate this test, please ignore this email.</em>
        </p>
    </body>
    </html>
    """

    try:
        email_notification = EmailNotification(config=config)
        email_notification.send()
        msg = "✅ Test email sent successfully!"
        _log.info(msg)
        if verbose:
            print(msg)
        return True
    except NotificationSendError as e:
        msg = f"❌ Error sending test email: {e}"
        _log.error(msg)
        if e.last_exception:
            _log.debug(
                "Traceback:\n%s", "".join(traceback.format_exception(e.last_exception))
            )
        if verbose:
            print(msg)
        return False


def format_email_preview(
    subject: str,
    body_html: str,
    attachments: list[str] | None = None,
) -> str:
    """Wrap a pre-built HTML email body with a minimal structure for previewing.

    Args:
        subject: The subject line of the email.
        body_html: The formatted HTML body.
        attachments: Optional list of attachment filenames to display (names only).

    Returns:
        Full HTML string.
    """
    attachment_section = ""
    if attachments:
        items = "".join(
            f"<li>{html.escape(os.path.basename(a))}</li>" for a in attachments
        )
        attachment_section = f"""
        <h3>Attachments:</h3>
        <ul>{items}</ul>
        """

    return f"""
    <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    padding: 20px;
                    color: #333;
                }}
                .subject {{
                    font-size: 1.5em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="subject">{html.escape(subject)}</div>
            {body_html}
            {attachment_section}
        </body>
    </html>
    """


def validate_email_config(
    config: EmailConfig,
    check_connectivity: bool = False,
    check_attachments: bool = False,
) -> list[str]:
    """Validate the EmailConfig for completeness and optional connectivity.

    Args:
        config: An EmailConfig instance.
        check_connectivity: If True, tests SMTP login credentials.
        check_attachments: If True, verifies that each attachment exists.

    Returns:
        A list of error messages. If empty, the config is considered valid.
    """
    errors: list[str] = []

    # validate required fields using metadata
    for field_def in fields(config):
        if field_def.metadata.get("required", False):
            value = getattr(config, field_def.name)
            if not value:
                errors.append(f"{field_def.name} is required and cannot be empty.")

    # validate recipient addresses
    if config.recipients:
        invalid_recipients = [r for r in config.recipients if not EMAIL_REGEX.match(r)]
        if invalid_recipients:
            errors.append(f"Invalid recipient email addresses: {invalid_recipients}")

    # validate attachment existence
    if check_attachments and config.attachments:
        for f in config.attachments:
            if not os.path.isfile(f):
                errors.append(f"Attachment not found: {f}")

    # test SMTP connection if requested
    if check_connectivity:
        try:
            with smtplib.SMTP(
                config.smtp_server, config.smtp_port, timeout=5
            ) as server:
                server.starttls()
                server.login(config.smtp_user, config.smtp_password)
        except Exception as e:  # pylint: disable=broad-except
            errors.append(f"SMTP connection failed: {e}")

    return errors
