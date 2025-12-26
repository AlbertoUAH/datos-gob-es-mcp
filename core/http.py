"""HTTP client with integrated logging and rate limiting."""

import time
from typing import Any

import httpx

from .logging import get_logger
from .ratelimit import RateLimiter

logger = get_logger("http")


class HTTPClientError(Exception):
    """Exception raised for HTTP client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class HTTPClient:
    """Async HTTP client with logging and rate limiting.

    This client wraps httpx and adds:
    - Automatic rate limiting per API
    - Structured logging of all requests
    - Consistent error handling

    Example:
        client = HTTPClient("datos.gob.es", "https://datos.gob.es/apidata/")
        response = await client.get("catalog/dataset.json")
    """

    def __init__(
        self,
        api_name: str,
        base_url: str,
        timeout: float = 30.0,
        rate_limit: bool = True,
    ):
        """Initialize HTTP client.

        Args:
            api_name: Name of the API for logging and rate limiting.
            base_url: Base URL for all requests.
            timeout: Request timeout in seconds.
            rate_limit: Whether to apply rate limiting.
        """
        self.api_name = api_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit = rate_limit

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"

    async def request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
        raise_for_status: bool = True,
    ) -> httpx.Response:
        """Make an HTTP request with logging and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: URL endpoint (relative to base_url or absolute)
            params: Query parameters
            headers: Request headers
            json_data: JSON body data
            raise_for_status: Whether to raise exception on HTTP errors

        Returns:
            httpx.Response object

        Raises:
            HTTPClientError: On request failure or HTTP error (if raise_for_status=True)
        """
        url = self._build_url(endpoint)

        # Apply rate limiting
        if self.rate_limit:
            await RateLimiter.acquire(self.api_name)

        # Log request start (DEBUG level to avoid noise)
        logger.debug(
            "request_start",
            api=self.api_name,
            method=method,
            url=url,
            params=params,
        )

        start_time = time.perf_counter()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    json=json_data,
                )

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Log successful request
                logger.info(
                    "request_complete",
                    api=self.api_name,
                    method=method,
                    url=url,
                    status=response.status_code,
                    duration_ms=round(duration_ms, 2),
                )

                # Raise for HTTP errors if requested
                if raise_for_status:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        raise HTTPClientError(
                            f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                            status_code=e.response.status_code,
                        ) from e

                return response

            except httpx.TimeoutException as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    "request_timeout",
                    api=self.api_name,
                    method=method,
                    url=url,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                )
                raise HTTPClientError(f"Request timed out: {e}") from e

            except httpx.RequestError as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    "request_failed",
                    api=self.api_name,
                    method=method,
                    url=url,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                )
                raise HTTPClientError(f"Request failed: {e}") from e

    async def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        raise_for_status: bool = True,
    ) -> httpx.Response:
        """Make a GET request."""
        return await self.request(
            "GET", endpoint, params=params, headers=headers,
            raise_for_status=raise_for_status
        )

    async def post(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
        raise_for_status: bool = True,
    ) -> httpx.Response:
        """Make a POST request."""
        return await self.request(
            "POST", endpoint, params=params, headers=headers,
            json_data=json_data, raise_for_status=raise_for_status
        )

    async def get_json(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make a GET request and return JSON response.

        Returns:
            Parsed JSON response.

        Raises:
            HTTPClientError: On request failure or JSON decode error.
        """
        response = await self.get(endpoint, params=params, headers=headers)
        try:
            return response.json()
        except Exception as e:
            raise HTTPClientError(f"Failed to parse JSON response: {e}") from e
