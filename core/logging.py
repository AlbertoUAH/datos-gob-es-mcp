"""Structured logging configuration using structlog."""

import logging
import os
import sys

import structlog


def setup_logging(
    level: str | None = None,
    json_format: bool | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var or INFO.
        json_format: If True, output JSON format. Defaults to LOG_FORMAT env var == 'json'.
    """
    # Get config from env vars if not provided
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if json_format is None:
        json_format = os.getenv("LOG_FORMAT", "console").lower() == "json"

    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )

    # Build processor chain
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with the given name.

    Args:
        name: Logger name (typically module name like 'http', 'ine', etc.)

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)


# Module-level logger
_root_logger = None


def _get_root_logger() -> structlog.stdlib.BoundLogger:
    """Get or create the root logger."""
    global _root_logger
    if _root_logger is None:
        _root_logger = get_logger("datos_gob_es")
    return _root_logger
