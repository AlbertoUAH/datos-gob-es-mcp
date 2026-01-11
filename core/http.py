"""HTTP client with integrated logging, rate limiting, connection pooling, and retry."""

import asyncio
import time
from typing import Any, ClassVar

import httpx

from .config import (
    HTTP2_ENABLED,
    HTTP_DEFAULT_TIMEOUT,
    HTTP_MAX_RETRIES,
    HTTP_POOL_MAX_CONNECTIONS,
    HTTP_POOL_MAX_KEEPALIVE,
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_RETRY_BACKOFF_INITIAL,
    HTTP_RETRY_BACKOFF_MAX,
)
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
    """Async HTTP client with logging, rate limiting, and connection pooling.

    This client wraps httpx and adds:
    - Connection pooling for better performance (reuses connections)
    - HTTP/2 support for multiplexing
    - Automatic rate limiting per API
    - Structured logging of all requests
    - Gzip compression support
    - Consistent error handling

    Example:
        client = HTTPClient("datos.gob.es", "https://datos.gob.es/apidata/")
        response = await client.get("catalog/dataset.json")
    """

    # Class-level connection pools (shared across instances with same base_url)
    _pools: ClassVar[dict[str, httpx.AsyncClient]] = {}

    @classmethod
    def _get_pool_key(cls, base_url: str) -> str:
        """Get pool key from base URL (normalize)."""
        return base_url.rstrip("/")

    @classmethod
    def _get_pool(cls, base_url: str, timeout: float) -> httpx.AsyncClient:
        """Get or create connection pool for base URL.

        Connection pools are reused across HTTPClient instances to maximize
        connection reuse and reduce latency.
        """
        pool_key = cls._get_pool_key(base_url)

        if pool_key not in cls._pools:
            cls._pools[pool_key] = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=HTTP_POOL_MAX_KEEPALIVE,
                    max_connections=HTTP_POOL_MAX_CONNECTIONS,
                ),
                http2=HTTP2_ENABLED,
                headers={
                    "Accept-Encoding": "gzip, deflate",
                },
            )
            logger.debug(
                "connection_pool_created",
                base_url=pool_key,
                max_keepalive=HTTP_POOL_MAX_KEEPALIVE,
                max_connections=HTTP_POOL_MAX_CONNECTIONS,
                http2=HTTP2_ENABLED,
            )

        return cls._pools[pool_key]

    @classmethod
    async def close_all_pools(cls) -> None:
        """Close all connection pools (for cleanup on shutdown)."""
        for pool_key, pool in list(cls._pools.items()):
            try:
                await pool.aclose()
                logger.debug("connection_pool_closed", base_url=pool_key)
            except Exception as e:
                logger.warning("connection_pool_close_error", base_url=pool_key, error=str(e))
        cls._pools.clear()

    @classmethod
    async def close_pool(cls, base_url: str) -> None:
        """Close a specific connection pool."""
        pool_key = cls._get_pool_key(base_url)
        if pool_key in cls._pools:
            try:
                await cls._pools[pool_key].aclose()
                del cls._pools[pool_key]
                logger.debug("connection_pool_closed", base_url=pool_key)
            except Exception as e:
                logger.warning("connection_pool_close_error", base_url=pool_key, error=str(e))

    def __init__(
        self,
        api_name: str,
        base_url: str,
        timeout: float = HTTP_DEFAULT_TIMEOUT,
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
        # Ensure pool exists for this base_url
        self._pool = self._get_pool(self.base_url, timeout)

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"

    def _should_retry(self, status_code: int | None, exception: Exception | None) -> bool:
        """Determine if a request should be retried.

        Retries on:
        - 5xx server errors (502, 503, 504, 500)
        - Timeout exceptions
        - Connection errors
        """
        if status_code is not None:
            return status_code >= 500
        if exception is not None:
            return isinstance(exception, (httpx.TimeoutException, httpx.ConnectError))
        return False

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = HTTP_RETRY_BACKOFF_INITIAL * (HTTP_RETRY_BACKOFF_FACTOR**attempt)
        return min(delay, HTTP_RETRY_BACKOFF_MAX)

    async def request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
        raise_for_status: bool = True,
        max_retries: int | None = None,
    ) -> httpx.Response:
        """Make an HTTP request with logging, rate limiting, connection reuse, and retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: URL endpoint (relative to base_url or absolute)
            params: Query parameters
            headers: Request headers
            json_data: JSON body data
            raise_for_status: Whether to raise exception on HTTP errors
            max_retries: Override default max retries (None uses HTTP_MAX_RETRIES)

        Returns:
            httpx.Response object

        Raises:
            HTTPClientError: On request failure or HTTP error (if raise_for_status=True)
        """
        url = self._build_url(endpoint)
        retries = max_retries if max_retries is not None else HTTP_MAX_RETRIES
        last_exception: Exception | None = None

        for attempt in range(retries + 1):
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
                attempt=attempt + 1,
            )

            start_time = time.perf_counter()

            try:
                # Use connection pool instead of creating new client each time
                response = await self._pool.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    json=json_data,
                )

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Check if we should retry on 5xx errors
                if self._should_retry(response.status_code, None) and attempt < retries:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        "request_retry",
                        api=self.api_name,
                        method=method,
                        url=url,
                        status=response.status_code,
                        attempt=attempt + 1,
                        max_retries=retries,
                        backoff_seconds=round(backoff, 2),
                    )
                    await asyncio.sleep(backoff)
                    continue

                # Log successful request
                logger.info(
                    "request_complete",
                    api=self.api_name,
                    method=method,
                    url=url,
                    status=response.status_code,
                    duration_ms=round(duration_ms, 2),
                    attempts=attempt + 1,
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
                last_exception = e

                if self._should_retry(None, e) and attempt < retries:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        "request_retry",
                        api=self.api_name,
                        method=method,
                        url=url,
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=retries,
                        backoff_seconds=round(backoff, 2),
                    )
                    await asyncio.sleep(backoff)
                    continue

                logger.error(
                    "request_timeout",
                    api=self.api_name,
                    method=method,
                    url=url,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                    attempts=attempt + 1,
                )
                raise HTTPClientError(f"Request timed out after {attempt + 1} attempts: {e}") from e

            except httpx.ConnectError as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                last_exception = e

                if self._should_retry(None, e) and attempt < retries:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        "request_retry",
                        api=self.api_name,
                        method=method,
                        url=url,
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=retries,
                        backoff_seconds=round(backoff, 2),
                    )
                    await asyncio.sleep(backoff)
                    continue

                logger.error(
                    "request_connect_error",
                    api=self.api_name,
                    method=method,
                    url=url,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                    attempts=attempt + 1,
                )
                raise HTTPClientError(
                    f"Connection failed after {attempt + 1} attempts: {e}"
                ) from e

            except httpx.RequestError as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    "request_failed",
                    api=self.api_name,
                    method=method,
                    url=url,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                    attempts=attempt + 1,
                )
                raise HTTPClientError(f"Request failed: {e}") from e

        # Should not reach here, but just in case
        raise HTTPClientError(
            f"Request failed after {retries + 1} attempts"
        ) from last_exception

    async def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        raise_for_status: bool = True,
    ) -> httpx.Response:
        """Make a GET request."""
        return await self.request(
            "GET", endpoint, params=params, headers=headers, raise_for_status=raise_for_status
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
            "POST",
            endpoint,
            params=params,
            headers=headers,
            json_data=json_data,
            raise_for_status=raise_for_status,
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
