"""Generic error handling utilities."""

import json

from .exceptions import APIClientError
from .logging import get_logger


def handle_api_error(
    e: Exception,
    context: str = "api_operation",
    logger_name: str = "api",
) -> str:
    """Format and log API error as JSON response.

    Args:
        e: The exception to handle.
        context: Context string for logging.
        logger_name: Name of the logger to use.

    Returns:
        JSON string with error details.
    """
    logger = get_logger(logger_name)
    logger.warning(f"{logger_name}_error", context=context, error=str(e))

    if isinstance(e, APIClientError):
        return json.dumps(
            {"error": e.message, "status_code": e.status_code}, ensure_ascii=False
        )

    return json.dumps({"error": str(e)}, ensure_ascii=False)
