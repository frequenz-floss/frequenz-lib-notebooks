# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Test module for notification_utils.py."""

from unittest.mock import MagicMock, patch

import pytest
from _pytest.logging import LogCaptureFixture

from frequenz.lib.notebooks import notification_utils
from frequenz.lib.notebooks.notification_service import (
    EmailConfig,
    NotificationSendError,
)


@pytest.fixture
def basic_email_config() -> EmailConfig:
    """Provide a valid EmailConfig for testing."""
    return EmailConfig(
        subject="Test Email",
        message="<p>This is a test message.</p>",
        recipients=["user@example.com"],
        smtp_server="smtp.test.com",
        smtp_port=587,
        smtp_user="user@test.com",
        smtp_password="password",
        from_email="noreply@test.com",
    )


# -------------------------------------------------------------------
# validate_email_config tests
# -------------------------------------------------------------------


def test_validate_email_config_passes_on_valid_input(
    basic_email_config: EmailConfig,
) -> None:
    """Test that no errors are returned for a valid config."""
    assert not notification_utils.validate_email_config(basic_email_config)


@pytest.mark.parametrize(
    "field_name, value, expected_error",
    [
        ("subject", "", "subject is required and cannot be empty."),
        ("message", "", "message is required and cannot be empty."),
        ("recipients", [], "recipients is required and cannot be empty."),
        ("smtp_server", "", "smtp_server is required and cannot be empty."),
        ("smtp_port", None, "smtp_port is required and cannot be empty."),
        ("smtp_user", "", "smtp_user is required and cannot be empty."),
        ("smtp_password", "", "smtp_password is required and cannot be empty."),
        ("from_email", "", "from_email is required and cannot be empty."),
    ],
)
def test_validate_email_config_field_errors(
    basic_email_config: EmailConfig,
    field_name: str,
    value: object,
    expected_error: str,
) -> None:
    """Test validate_email_config catches missing/invalid required fields."""
    setattr(basic_email_config, field_name, value)
    errors = notification_utils.validate_email_config(basic_email_config)

    if expected_error:
        assert expected_error in errors
    else:
        assert all(expected_error not in e for e in errors)


def test_validate_email_config_check_attachments_flag(
    basic_email_config: EmailConfig,
) -> None:
    """Test that missing attachments are caught only when check_attachments=True."""
    basic_email_config.attachments = ["non_existent_file.csv"]

    errors_with_check = notification_utils.validate_email_config(
        basic_email_config, check_attachments=True
    )
    errors_without_check = notification_utils.validate_email_config(
        basic_email_config, check_attachments=False
    )

    assert any("Attachment not found" in e for e in errors_with_check)
    assert all("Attachment not found" not in e for e in errors_without_check)


@patch("smtplib.SMTP")
def test_validate_email_config_check_connectivity_flag(
    mock_smtp: MagicMock,
    basic_email_config: EmailConfig,
) -> None:
    """Test that SMTP connection errors are caught only when check_connectivity=True."""
    mock_smtp.side_effect = Exception("Failed to connect")

    errors_with_check = notification_utils.validate_email_config(
        basic_email_config, check_connectivity=True
    )
    errors_without_check = notification_utils.validate_email_config(
        basic_email_config, check_connectivity=False
    )

    assert any("SMTP connection failed" in e for e in errors_with_check)
    assert all("SMTP connection failed" not in e for e in errors_without_check)


def test_validate_email_config_catches_invalid_recipients(
    basic_email_config: EmailConfig,
) -> None:
    """Test that improperly formatted email addresses are caught."""
    basic_email_config.recipients = ["bad-email", "also@bad", "nope.com"]
    errors = notification_utils.validate_email_config(basic_email_config)
    assert len(errors) == 1
    assert "Invalid recipient" in errors[0]


def test_validate_email_config_checks_missing_attachments(
    basic_email_config: EmailConfig,
) -> None:
    """Test that missing files in attachment list are reported."""
    basic_email_config.attachments = ["does_not_exist.txt"]
    errors = notification_utils.validate_email_config(
        basic_email_config, check_attachments=True
    )
    assert any("Attachment not found" in e for e in errors)


@patch("smtplib.SMTP")
def test_validate_email_config_smtp_connection_failure(
    mock_smtp: MagicMock, basic_email_config: EmailConfig
) -> None:
    """Test that SMTP connectivity errors are captured."""
    mock_smtp.side_effect = Exception("Boom!")
    errors = notification_utils.validate_email_config(
        basic_email_config, check_connectivity=True
    )
    assert any("SMTP connection failed" in e for e in errors)


# -------------------------------------------------------------------
# format_email_preview tests
# -------------------------------------------------------------------


def test_format_email_preview_structure_and_content() -> None:
    """Test that format_email_preview generates valid HTML with optional attachments."""
    subject = "Alert Summary"
    body_html = "<p>This is a test alert message.</p>"
    attachments = ["file1.csv", "log.txt"]

    html_output = notification_utils.format_email_preview(
        subject=subject,
        body_html=body_html,
        attachments=attachments,
    )

    assert "<html>" in html_output
    assert subject in html_output
    assert "file1.csv" in html_output
    assert "log.txt" in html_output
    assert "<ul>" in html_output


def test_format_email_preview_without_attachments() -> None:
    """Test that preview renders fine with no attachments provided."""
    html_output = notification_utils.format_email_preview(
        subject="Hello",
        body_html="<p>Body</p>",
    )
    assert "Attachments:" not in html_output
    assert "<p>Body</p>" in html_output


# -------------------------------------------------------------------
# send_test_email tests
# -------------------------------------------------------------------


@patch("frequenz.lib.notebooks.notification_service.EmailNotification.send")
def test_send_test_email_success(basic_email_config: EmailConfig) -> None:
    """Test that send_test_email logs success correctly."""
    success = notification_utils.send_test_email(basic_email_config)
    assert success is True


@patch("frequenz.lib.notebooks.notification_service.EmailNotification.send")
def test_send_test_email_handles_failure(
    mock_send: MagicMock,
    caplog: LogCaptureFixture,
    basic_email_config: EmailConfig,
) -> None:
    """Test that send_test_email logs error and traceback on failure."""
    cause = RuntimeError("timeout")
    exception = NotificationSendError("Retry failed", last_exception=cause)
    mock_send.side_effect = exception  # Important: patch `send`, not `send_with_retry`

    with caplog.at_level("DEBUG", logger="frequenz.lib.notebooks.notification_utils"):
        success = notification_utils.send_test_email(basic_email_config)

    assert success is False
    assert any("retry failed" in msg.lower() for msg in caplog.messages)
    assert any("traceback" in msg.lower() for msg in caplog.messages)
    assert any("timeout" in msg.lower() for msg in caplog.messages)
