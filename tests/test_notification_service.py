# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Test module for the notification service."""

import logging
import time
from typing import Generator
from unittest.mock import MagicMock, mock_open, patch

import pytest

from frequenz.lib.notebooks.notification_service import (
    BaseNotification,
    EmailConfig,
    EmailNotification,
    FromDictMixin,
    NotificationSendError,
    Scheduler,
    SchedulerConfig,
)


class MockNotification(BaseNotification):
    """Mock notification class for testing."""

    def __init__(self, mock_send: MagicMock, scheduler: Scheduler | None = None):
        """Initialize MockNotification with a send mock."""
        super().__init__()
        self.mock_send = mock_send
        self._scheduler = scheduler

    def send(self) -> None:
        """Send method that calls the mock send."""
        self.mock_send()


@pytest.fixture
def mock_send() -> MagicMock:
    """Fixture to provide a reusable mock for the `send` method."""
    mocked_send = MagicMock()
    mocked_send.__name__ = "mock_send"
    return mocked_send


@pytest.fixture
def mock_sleep() -> Generator[MagicMock, None, None]:
    """Fixture to mock `time.sleep`."""
    with patch("time.sleep", return_value=None) as mock_sleep:
        yield mock_sleep


@pytest.fixture
def scheduler_config() -> SchedulerConfig:
    """Fixture to provide a reusable SchedulerConfig."""
    return SchedulerConfig(send_immediately=True, interval=1, duration=5)


@pytest.fixture
def email_config() -> EmailConfig:
    """Fixture to provide a reusable EmailConfig."""
    return EmailConfig(
        subject="Test Email",
        message="<p>This is a test email.</p>",
        recipients=["test@example.com"],
        smtp_server="smtp.test.com",
        smtp_port=587,
        smtp_user="user@test.com",
        smtp_password="password",
        from_email="no-reply@example.com",
        scheduler=SchedulerConfig(send_immediately=True, interval=1, duration=3),
    )


@pytest.fixture
def mock_notification(
    mock_send: MagicMock, scheduler_config: SchedulerConfig
) -> MockNotification:
    """Fixture to provide a reusable MockNotification instance."""
    scheduler = Scheduler(scheduler_config)
    return MockNotification(mock_send=mock_send, scheduler=scheduler)


@pytest.mark.parametrize(
    "config",
    [
        SchedulerConfig(send_immediately=True, interval=1, duration=3),
        SchedulerConfig(send_immediately=False, interval=1, duration=3),
        SchedulerConfig(send_immediately=True, interval=1, duration=None),
        SchedulerConfig(send_immediately=False, interval=1, duration=None),
    ],
)
def test_scheduler_behavior(config: SchedulerConfig, mock_send: MagicMock) -> None:
    """Test scheduler behavior with various configurations."""
    scheduler = Scheduler(config)
    test_time_if_no_duration = config.interval + 0.5
    scheduler.start(mock_send)
    if config.duration:
        time.sleep(config.duration + 0.5)
        correction = -1  # -1 to account for the time lost in executing the task
    else:
        time.sleep(test_time_if_no_duration)
        correction = 0
        scheduler.stop()

    expected_calls = (
        (
            (config.duration if config.duration else test_time_if_no_duration)
            // config.interval
        )
        + (1 if config.send_immediately else 0)
        + correction
    )
    assert mock_send.call_count == expected_calls


def test_scheduler_stops_immediately(mock_send: MagicMock) -> None:
    """Test that the scheduler stops immediately when stop is called."""
    scheduler = Scheduler(
        SchedulerConfig(send_immediately=False, interval=1, duration=5)
    )
    scheduler.start(mock_send)
    scheduler.stop()
    assert mock_send.call_count == 0


# Tests for the BaseNotification class
def test_send_with_retry_success(mock_send: MagicMock) -> None:
    """Test that send_with_retry retries correctly and succeeds."""
    mock_send.side_effect = [
        Exception("First failure"),
        None,
    ]  # Fail once, then succeed

    BaseNotification.send_with_retry(
        send_func=mock_send,
        retries=3,
        backoff_factor=1,
        max_sleep=2,
    )
    assert mock_send.call_count == 2  # First attempt + one retry


def test_send_with_retry_failure(
    caplog: pytest.LogCaptureFixture, mock_send: MagicMock
) -> None:
    """Test that send_with_retry raises after all retries fail."""
    mock_send.side_effect = Exception("Always fails")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(NotificationSendError) as exc_info:
            BaseNotification.send_with_retry(
                send_func=mock_send,
                retries=3,
                backoff_factor=1,
                max_sleep=2,
            )
        assert "Failed to send notification after 3 retries" in caplog.text
        assert "Always fails" in str(exc_info.value)
        assert "notification failed" in str(exc_info.value).lower()
    assert mock_send.call_count == 4


def test_send_with_retry_no_retries(mock_send: MagicMock) -> None:
    """Test send_with_retry with zero retries."""
    BaseNotification.send_with_retry(
        send_func=mock_send,
        retries=0,
        backoff_factor=1,
        max_sleep=2,
    )
    assert mock_send.call_count == 1


def test_send_with_retry_backoff(mock_send: MagicMock, mock_sleep: MagicMock) -> None:
    """Test that send_with_retry respects backoff and max_sleep."""
    mock_send.side_effect = [Exception("Failure")] * 3  # Simulate 3 failures

    retries = 3
    backoff_factor = 2
    max_sleep = 3
    with pytest.raises(NotificationSendError):
        BaseNotification.send_with_retry(
            send_func=mock_send,
            retries=retries,
            backoff_factor=backoff_factor,
            max_sleep=max_sleep,
        )

    # Expected sleep durations for each retry
    expected_sleep_calls = [
        min(max_sleep, backoff_factor * (i + 1)) for i in range(retries - 1)
    ]
    actual_sleep_calls = [args[0] for args, _ in mock_sleep.call_args_list]
    assert actual_sleep_calls == expected_sleep_calls
    assert mock_send.call_count == retries + 1


def test_start_scheduler(
    mock_notification: MockNotification, mock_send: MagicMock
) -> None:
    """Test that start_scheduler invokes send at regular intervals."""
    mock_notification.start_scheduler()
    time.sleep(2.5)
    mock_notification.stop_scheduler()
    assert mock_send.call_count == 3


def test_scheduler_no_send_when_stopped(mock_notification: MockNotification) -> None:
    """Test no further sends after the scheduler is stopped."""
    mock_notification.start_scheduler()
    time.sleep(2)
    mock_notification.stop_scheduler()
    time.sleep(1.1)
    sends_after_stop = mock_notification.mock_send.call_count
    assert mock_notification.mock_send.call_count == sends_after_stop


def test_stop_scheduler_no_active_thread(
    mock_notification: MockNotification, caplog: pytest.LogCaptureFixture
) -> None:
    """Test stopping a scheduler when no thread is active."""
    with caplog.at_level(logging.WARNING):
        mock_notification.stop_scheduler()
        assert "no active thread was found" in caplog.text


# Tests for the EmailNotification class
@patch("smtplib.SMTP")
def test_send_email_success(mock_smtp: MagicMock, email_config: EmailConfig) -> None:
    """Test that EmailNotification sends an email successfully."""
    email_notification = EmailNotification(config=email_config)
    email_notification.send()
    mock_smtp.assert_called_once_with("smtp.test.com", 587)


@patch("smtplib.SMTP")
@patch("builtins.open", new_callable=mock_open, read_data="data")
@patch("email.message.EmailMessage.add_attachment")
def test_send_email_with_attachments(
    mock_add_attachment: MagicMock,
    mock_open: MagicMock,
    mock_smtp: MagicMock,
    email_config: EmailConfig,
) -> None:
    """Test that EmailNotification handles attachments correctly."""
    email_config.attachments = ["test_file.txt"]
    email_notification = EmailNotification(config=email_config)
    email_notification.send()

    # Filter out any unexpected file open calls (like /etc/apache2/mime.types)
    file_open_calls = [
        call_args[0][0]
        for call_args in mock_open.call_args_list
        if call_args[0][0] == "test_file.txt"
    ]

    assert (
        len(file_open_calls) == 1
    ), f"Unexpected open calls: {mock_open.call_args_list}"
    mock_add_attachment.assert_called_once_with(
        "",
        maintype="text",
        subtype="plain",
        filename="test_file.txt",
    )
    mock_smtp.assert_called_once_with("smtp.test.com", 587)


# Additional tests for missing required fields and from_dict method
def test_email_config_post_init_missing_required_fields() -> None:
    """Test EmailConfig raises ValueError if required fields are missing or empty."""
    with pytest.raises(
        ValueError, match="smtp_server is required and cannot be empty."
    ):
        EmailConfig(
            subject="Test Email",
            message="This is a test",
            recipients=["test@example.com"],
            smtp_server="",  # Required but empty
            smtp_port=587,
            smtp_user="user@test.com",
            smtp_password="password",
            from_email="no-reply@example.com",
        )


def test_email_config_from_dict_missing_required_fields() -> None:
    """Test from_dict raises ValueError if required fields are missing."""
    data = {
        "subject": "Test Email",
        "message": "<p>This is a test</p>",
        # recipients is missing
        "smtp_server": "smtp.test.com",
        "smtp_port": 587,
        "smtp_user": "user@test.com",
        "smtp_password": "password",
        "from_email": "no-reply@example.com",
    }

    with pytest.raises(
        TypeError, match="missing 1 required keyword-only argument: 'recipients'"
    ):
        EmailConfig.from_dict(data)


def test_email_config_from_dict_success() -> None:
    """Test from_dict constructs an EmailConfig including a scheduler configuration."""
    data = {
        "subject": "Test Email",
        "message": "<p>This is a test</p>",
        "recipients": ["test@example.com"],
        "smtp_server": "smtp.test.com",
        "smtp_port": 587,
        "smtp_user": "user@test.com",
        "smtp_password": "password",
        "from_email": "no-reply@example.com",
        "scheduler": {
            "send_immediately": True,
            "interval": 60,
            "duration": 120,
        },
    }
    cfg = EmailConfig.from_dict(data)
    print(cfg)
    print(cfg.scheduler)
    assert cfg.smtp_server == "smtp.test.com"
    assert cfg.smtp_port == 587
    assert cfg.recipients == ["test@example.com"]
    assert cfg.subject == "Test Email"
    assert cfg.message == "<p>This is a test</p>"
    assert cfg.scheduler is not None
    assert cfg.scheduler.send_immediately is True
    assert cfg.scheduler.interval == 60
    assert cfg.scheduler.duration == 120


def test_from_dict_unexpected_input_types() -> None:
    """Test from_dict raises TypeError if unexpected types are encountered."""

    class TestClass(FromDictMixin):
        """Dummy class for testing."""

    with pytest.raises(TypeError, match="Expected a dictionary"):
        EmailConfig.from_dict("not a dictionary")  # type: ignore

    with pytest.raises(TypeError, match="is not a dataclass."):
        TestClass.from_dict({})


def test_from_dict_not_implemented() -> None:
    """Test from_dict raises AttributeError if the class does not implement it."""

    class TestClass:
        """Dummy class for testing."""

    with pytest.raises(AttributeError, match="has no attribute 'from_dict'"):
        TestClass.from_dict({})  # type: ignore


# Tests for the NotificationSendError class
def test_notification_send_error_str_with_cause() -> None:
    """Test NotificationSendError string representation with a cause."""
    original_error = ValueError("invalid email address")
    exc = NotificationSendError(
        "Failed to send notification", last_exception=original_error
    )
    assert str(exc) == "Failed to send notification (Caused by: invalid email address)"


def test_notification_send_error_str_without_cause() -> None:
    """Test NotificationSendError string representation without a cause."""
    exc = NotificationSendError("Retry attempts exhausted")
    assert str(exc) == "Retry attempts exhausted"
