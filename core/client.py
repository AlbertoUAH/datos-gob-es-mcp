"""Base API client class."""

from typing import Any, ClassVar

from .config import HTTP_DEFAULT_TIMEOUT
from .exceptions import APIClientError
from .http import HTTPClient


class BaseAPIClient:
    """Base class for API clients with common functionality.

    Subclasses should define class attributes:
        - BASE_URL: str - The base URL for the API
        - API_NAME: str - Name for logging and rate limiting
        - ERROR_CLASS: type - Exception class to raise on errors

    Example:
        class MyClient(BaseAPIClient):
            BASE_URL = "https://api.example.com/"
            API_NAME = "example"
            ERROR_CLASS = MyClientError

            async def get_data(self) -> dict:
                return await self._request("data/endpoint")
    """

    BASE_URL: ClassVar[str] = ""
    API_NAME: ClassVar[str] = "api"
    ERROR_CLASS: ClassVar[type] = APIClientError

    def __init__(self, timeout: float = HTTP_DEFAULT_TIMEOUT):
        """Initialize the API client.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.http = HTTPClient(self.API_NAME, self.BASE_URL, timeout)

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make an HTTP request to the API.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            headers: Additional headers.

        Returns:
            Parsed JSON response.

        Raises:
            APIClientError: On request failure.
        """
        try:
            return await self.http.get_json(endpoint, params=params, headers=headers)
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            raise self.ERROR_CLASS(str(e), status_code=status_code) from e
