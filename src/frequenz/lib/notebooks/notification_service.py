# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
This module provides a notification service for sending alert notifications.

The service supports sending email with optional attachments. It also provides
a scheduler for sending periodic notifications with configurable intervals and
durations. The service is designed to handle retries and backoff for failed
notification attempts.


Example usage
=============
# Configuration for email notification
email_config = EmailConfig(
    subject="Critical Alert",
    message="Inverter is in error mode",
    recipients=["recipient@example.com"],
    smtp_server="smtp.example.com",
    smtp_port=587,
    smtp_user="user@example.com",
    smtp_password="password",
    from_email="alert@example.com",
    attachments=["alert_records.csv"]
    scheduler=SchedulerConfig(
        send_immediately=True,
        interval=60,
        duration=3600,
    ),
)

email_config_dict = {
    "subject": "Critical Alert",
    "message": "Inverter is in error mode",
    "recipients": ["recipient@example.com"],
    "smtp_server": "smtp.example.com",
    "smtp_port": 587,
    "smtp_user": "user@example.com",
    "smtp_password": "password",
    "from_email": "alert@example.com",
    "attachments": ["alert_records.csv"],
    "scheduler": {
        "send_immediately": True,
        "interval": 60,
        "duration": 3600,
    },
}

# Create notification objects
email_notification = EmailNotification(config=email_config)
email_notification_2 = EmailConfig.from_dict(email_config_dict)

# Send one-off notification
email_notification.send()

# Start periodic notifications
email_notification.start_scheduler()

# Stop the scheduler after some time if needed
time.sleep(300)
email_notification.stop_scheduler()
"""

import logging
import mimetypes
import os
import smtplib
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from email.message import EmailMessage
from smtplib import SMTPException
from types import UnionType
from typing import Any, Callable, TypeVar, Union, get_args, get_origin

_log = logging.getLogger(__name__)


DataclassT = TypeVar("DataclassT", bound="FromDictMixin")


class FromDictMixin:
    """A mixin to add a from_dict class method for dataclasses."""

    @classmethod
    def from_dict(cls: type[DataclassT], data: dict[str, Any]) -> DataclassT:
        """Create an instance of the dataclass from a dictionary.

        This method handles:
            - Standard fields: Assigns values directly.
            - Nested dataclasses (that also inherit FromDictMixin): Recursively
                converts dictionaries into dataclass instances.
            - Optional fields with union types: Extracts the dataclass type from
                the union if present and handles None values.
            - Type validation: Ensures the provided data matches expected field
                types.

        Args:
            data: The data dictionary to be mapped to the dataclass.

        Returns:
            An instance of the dataclass.

        Raises:
            TypeError: If the input data is not a dictionary or `cls` is not a
                dataclass.
        """

        def is_union(t: Any) -> bool:
            """Check if a type is a Union."""
            return isinstance(t, UnionType) or get_origin(t) is Union

        if not isinstance(data, dict):
            raise TypeError(
                f"Expected a dictionary to create {cls.__name__}, got {type(data)}."
            )

        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass.")

        field_types = {f.name: f.type for f in fields(cls)}
        init_kwargs = {}

        for key, value in data.items():
            if key not in field_types:
                continue
            field_type = field_types[key]

            # handle union types (e.g., SchedulerConfig | None or Union[SchedulerConfig, None])
            if is_union(field_type):
                if value is None:
                    init_kwargs[key] = None
                    continue

                # find a dataclass type if one exists
                field_type_args = get_args(field_type)
                for arg in field_type_args:
                    if (
                        arg is not type(None)
                        and is_dataclass(arg)
                        and issubclass(arg, FromDictMixin)
                    ):
                        field_type = arg
                        break

            # if field is a nested dataclass implementing FromDictMixin and the value is a dict
            if (
                is_dataclass(field_type)
                and isinstance(value, dict)
                and issubclass(field_type, FromDictMixin)
            ):
                init_kwargs[key] = field_type.from_dict(value)
            else:
                init_kwargs[key] = value

        instance = cls(**init_kwargs)
        return instance


@dataclass
class SchedulerConfig(FromDictMixin):
    """Configuration for the scheduler."""

    send_immediately: bool = field(
        default=False,
        metadata={
            "description": (
                "Whether to send the first notification immediately "
                "upon starting the scheduler or after the first interval"
            ),
        },
    )

    interval: int = field(
        default=60,
        metadata={
            "description": (
                "Frequency in seconds to send the notification if the "
                "scheduler is enabled"
            ),
            "validate": lambda x: x > 0,
        },
    )

    duration: int | None = field(
        default=None,
        metadata={
            "description": (
                "Total duration in seconds to run the scheduler. If None, it runs "
                "indefinitely"
            ),
            "validate": lambda x: x is None or x > 0,
        },
    )


@dataclass
class BaseNotificationConfig(FromDictMixin):
    """Base configuration for notifications."""

    subject: str = field(
        metadata={
            "description": "Subject or title of the notification",
            "required": True,
        },
    )

    message: str = field(
        metadata={
            "description": "Message content of the notification",
            "required": True,
        },
    )

    retries: int = field(
        default=3,
        metadata={
            "description": "Number of retry attempts after the first failure",
            "validate": lambda x: 1 < x <= 10,
        },
    )

    backoff_factor: int = field(
        default=3,
        metadata={
            "description": "Delay factor for backoff calculation",
            "validate": lambda x: x > 0,
        },
    )

    max_retry_sleep: int = field(
        default=30,
        metadata={
            "description": (
                "Maximum sleep time between retries in seconds unless a scheduler "
                "is used in which case it is capped at the minimum of the interval "
                "and this value"
            ),
            "validate": lambda x: 0 < x <= 60,
        },
    )

    attachments: list[str] | None = field(
        default=None,
        metadata={
            "description": "List of files to attach to the notification",
        },
    )

    scheduler: SchedulerConfig | None = field(
        default=None,
        metadata={
            "description": "Configuration for the scheduler",
        },
    )


@dataclass(kw_only=True)
class EmailConfig(BaseNotificationConfig):
    """Configuration for sending email notifications."""

    smtp_server: str = field(
        metadata={
            "description": "SMTP server address",
            "required": True,
        },
    )

    smtp_port: int = field(
        metadata={
            "description": "SMTP server port",
            "required": True,
        },
    )

    smtp_user: str = field(
        metadata={
            "description": "SMTP server username",
            "required": True,
        },
    )

    smtp_password: str = field(
        metadata={
            "description": "SMTP server password",
            "required": True,
        },
    )

    from_email: str = field(
        metadata={
            "description": "Email address of the sender",
            "required": True,
        },
    )

    recipients: list[str] = field(
        metadata={
            "description": "List of email addresses as recipients",
            "required": True,
        },
    )

    def __post_init__(self) -> None:
        """Validate required fields that must not be empty."""
        if not self.smtp_server:
            raise ValueError("smtp_server is required and cannot be empty.")
        if not self.smtp_port:
            raise ValueError("smtp_port is required and cannot be empty.")
        if not self.smtp_user:
            raise ValueError("smtp_user is required and cannot be empty.")
        if not self.smtp_password:
            raise ValueError("smtp_password is required and cannot be empty.")
        if not self.from_email:
            raise ValueError("from_email is required and cannot be empty.")
        if not self.recipients:
            raise ValueError("recipients is required and cannot be empty.")


class Scheduler:
    """Utility class for scheduling periodic tasks."""

    def __init__(self, config: SchedulerConfig) -> None:
        """Initialise the scheduler.

        Args:
            config: Configuration for the scheduler.
        """
        self._config = config
        self.task: Callable[..., None] | None = None
        self._task_name: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._time_awoke: float = 0.0  # time when the scheduler awoke from sleep

    def start(self, task: Callable[..., None], **kwargs: Any) -> None:
        """Start the scheduler for a given task.

        Args:
            task: The task to execute periodically.
            **kwargs: Arguments to pass to the task.
        """
        self.task = task
        self._task_name = task.__name__
        _log.info(
            "Starting scheduler for task '%s' to execute every %d seconds and %s",
            self._task_name,
            self._config.interval,
            (
                f"for {self._config.duration} seconds"
                if self._config.duration
                else "indefinitely"
            ),
        )
        self._thread = threading.Thread(
            target=self._run_task, args=(kwargs,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._thread is not None:
            if self._thread.is_alive():
                _log.info("Stopping scheduler for %s", self._task_name)
                self._stop_event.set()
                if not self._stop_event.is_set():
                    _log.error("Failed to stop scheduler for %s", self._task_name)
        else:
            _log.warning(
                "Attempted to stop scheduler for %s, but no active thread was found.",
                self._task_name,
            )
        _log.info("Scheduler successfully stopped")

    def _run_task(self, kwargs: dict[str, Any]) -> None:
        """Run the scheduled task.

        Args:
            kwargs: Arguments to pass to the task.
        """
        start_time = time.time()
        if self._config.send_immediately:
            self._execute_task(kwargs)
        else:
            _log.info(
                "Waiting for first interval before sending the first notification."
            )
            self._stop_event.wait(self._config.interval)
            self._time_awoke = time.time()

        while not self._stop_event.is_set():
            if self._should_stop(start_time):
                break
            self._execute_task(kwargs)

    def _should_stop(self, start_time: float) -> bool:
        """Determine if the scheduler should stop.

        Args:
            start_time: The time the scheduler started.

        Returns:
            True if the scheduler should stop, False otherwise.
        """
        if (
            self._config.duration is not None
            and (time.time() - self._time_awoke - start_time) >= self._config.duration
        ):
            return True
        return False

    def _execute_task(self, kwargs: dict[str, Any]) -> None:
        """Execute the scheduled task and handle interval waiting.

        Args:
            kwargs: Arguments to pass to the task.
        """
        task_start_time = time.time()
        try:
            if self.task:
                self.task(**kwargs)
        except Exception as e:  # pylint: disable=broad-except
            _log.error(
                "Error occurred during scheduled execution of %s: %s",
                self._task_name,
                e,
            )
        finally:
            task_elapsed = time.time() - task_start_time
            sleep_duration = max(0, self._config.interval - task_elapsed)
            _log.info(
                "Scheduled execution completed for %s. Sleeping for %d seconds.",
                self._task_name,
                sleep_duration,
            )
            self._stop_event.wait(sleep_duration)
            self._time_awoke = time.time()


class BaseNotification:
    """Base class for all notification types.

    Subclasses must implement the `send` method.
    """

    def __init__(self) -> None:
        """Initialise the notification object."""
        self._scheduler: Scheduler | None = None

    @staticmethod
    def send_with_retry(
        *,
        send_func: Callable[..., None],
        retries: int,
        backoff_factor: int,
        max_sleep: int,
        **kwargs: Any,
    ) -> None:
        """Attempt to execute the `send_func` with retries and backoff.

        Args:
            send_func: The function to execute (e.g., send_email_alert).
            retries: Number of retry attempts after the first failure.
            backoff_factor: Delay factor for (linear) backoff calculation.
            max_sleep: Maximum sleep time in seconds.
            **kwargs: Keyword arguments for the send_func.

        Raises:
            NotificationSendError: If the notification fails after all retry attempts.
        """
        for attempt in range(retries + 1):
            try:
                send_func(**kwargs)
                _log.info("Successfully sent notification on attempt %d", attempt + 1)
                return
            except Exception as e:  # pylint: disable=broad-except
                last_exception = e
                _log.error("Attempt %d failed: %s", attempt + 1, e)
                if attempt < retries - 1:
                    linear_backoff = backoff_factor * (attempt + 1)
                    time.sleep(min(max_sleep, linear_backoff))
        _log.error("Failed to send notification after %d retries", retries)
        raise NotificationSendError(
            "Notification failed after all retry attempts.",
            last_exception=last_exception,
        )

    def start_scheduler(self) -> None:
        """Start the scheduler if configured."""
        if self._scheduler:
            _log.info("Starting scheduler for %s", self.__class__.__name__)
            self._scheduler.start(self.send)
        else:
            _log.warning("No scheduler config provided. Cannot start scheduler.")

    def stop_scheduler(self) -> None:
        """Stop the running scheduler."""
        if not self._scheduler:
            _log.warning("No active scheduler to stop.")
            return
        _log.info("Stopping scheduler for notification: %s", self.__class__.__name__)
        self._scheduler.stop()

    @abstractmethod
    def send(self) -> None:
        """Send the notification. To be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement the send method.")


class EmailNotification(BaseNotification):
    """Handles email notifications.

    This class sends HTML emails with optional attachments. It uses the smtplib
    library to connect to an SMTP server and send the email.
    """

    def __init__(self, config: EmailConfig) -> None:
        """Initialise the email notification with configuration.

        Args:
            config: Configuration for email notifications.
        """
        super().__init__()
        self._config: EmailConfig = config

    def send(self) -> None:
        """Send the email notification."""
        self.send_with_retry(
            send_func=self._send_email,
            retries=self._config.retries,
            backoff_factor=self._config.backoff_factor,
            max_sleep=(
                self._config.scheduler.interval
                if self._config.scheduler
                else self._config.max_retry_sleep
            ),
            config=self._config,
        )

    def _send_email(
        self,
        config: EmailConfig,
    ) -> None:
        """Send an HTML email alert with optional attachments.

        Args:
            config: Email configuration object.
        """
        msg = EmailMessage()
        msg["From"] = config.from_email
        msg["To"] = ", ".join(config.recipients)
        msg["Subject"] = config.subject
        msg.add_alternative(config.message, subtype="html")

        if config.attachments:
            self._attach_files(msg, config.attachments)

        smtp_settings: dict[str, str | int] = {
            "server": config.smtp_server,
            "port": config.smtp_port,
            "user": config.smtp_user,
            "password": config.smtp_password,
        }
        self._connect_and_send(msg, smtp_settings, config.recipients)

    def _attach_files(self, msg: EmailMessage, attachments: list[str]) -> None:
        """Attach files to the email.

        Args:
            msg: EmailMessage object.
            attachments: List of file paths to attach.
        """
        failed_attachments = []
        for file in attachments:
            try:
                with open(file, "rb") as f:
                    maintype, subtype = self._get_mime_type(file)
                    msg.add_attachment(
                        f.read(),
                        maintype=maintype,
                        subtype=subtype,
                        filename=os.path.basename(file),
                    )
            except OSError as e:
                failed_attachments.append(file)
                _log.error("Failed to attach file %s: %s", file, e)
        if failed_attachments:
            _log.warning(
                "The following attachments could not be added: %s", failed_attachments
            )

    @staticmethod
    def _get_mime_type(file: str) -> tuple[str, str]:
        """Determine the MIME type of a file with fallback.

        Args:
            file: Path to the file.

        Returns:
            A tuple containing the MIME type (maintype, subtype).
        """
        mime_type, _ = mimetypes.guess_type(file)
        if mime_type:
            maintype, subtype = mime_type.split("/")
        else:
            # generic fallback logic
            if file.endswith((".csv", ".txt", ".log")):
                maintype, subtype = "text", "plain"
            elif file.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                maintype, subtype = "image", "png"
            else:
                # default: binary file fallback
                maintype, subtype = "application", "octet-stream"
        return maintype, subtype

    @staticmethod
    def _connect_and_send(
        msg: EmailMessage, smtp_settings: dict[str, str | int], to_emails: list[str]
    ) -> None:
        """Send the email via SMTP.

        Args:
            msg: EmailMessage object containing the email content.
            smtp_settings: SMTP server configuration.
            to_emails: List of recipient email addresses.

        Raises:
            SMTPException: If the email fails to send.
        """
        try:
            with smtplib.SMTP(
                str(smtp_settings["server"]), int(smtp_settings["port"])
            ) as server:
                server.starttls()
                server.login(str(smtp_settings["user"]), str(smtp_settings["password"]))
                server.send_message(msg)
            _log.info("Email sent successfully to %s", to_emails)
        except SMTPException as e:
            _log.error("Failed to send email: %s", e)
            raise


class NotificationSendError(Exception):
    """Raised when sending a notification fails after all retry attempts."""

    def __init__(self, message: str, last_exception: Exception | None = None) -> None:
        """Initialise the error with a message and optional last exception.

        Args:
            message: Error message.
            last_exception: The last exception encountered during the send process.
        """
        super().__init__(message)
        self.last_exception = last_exception

    def __str__(self) -> str:
        """Return a string representation of the error."""
        base = super().__str__()
        if self.last_exception:
            return f"{base} (Caused by: {self.last_exception})"
        return base
