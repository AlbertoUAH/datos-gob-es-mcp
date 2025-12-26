"""Tests for the core module (logging, rate limiting, HTTP client)."""

import asyncio
import pytest
import respx
from httpx import Response
from unittest.mock import patch

from core.logging import setup_logging, get_logger
from core.ratelimit import RateLimiter, DEFAULT_LIMITS
from core.http import HTTPClient, HTTPClientError


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogging:
    """Tests for structured logging."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_with_level(self):
        """Test logging setup with custom level."""
        setup_logging(level="DEBUG")
        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_json_format(self):
        """Test logging setup with JSON format."""
        setup_logging(json_format=True)
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger_returns_bound_logger(self):
        """Test that get_logger returns a bound logger."""
        logger = get_logger("test_module")
        assert logger is not None
        # Should be able to call logging methods
        logger.info("test message")

    def test_logger_with_context(self):
        """Test logger with additional context."""
        logger = get_logger("test")
        logger.info("test_event", key="value", count=42)


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for the rate limiter."""

    def setup_method(self):
        """Reset rate limiters before each test."""
        RateLimiter.reset()

    def test_get_limiter_creates_new(self):
        """Test that get_limiter creates a new limiter."""
        limiter = RateLimiter.get_limiter("test_api")
        assert limiter is not None

    def test_get_limiter_reuses_existing(self):
        """Test that get_limiter reuses existing limiters."""
        limiter1 = RateLimiter.get_limiter("test_api")
        limiter2 = RateLimiter.get_limiter("test_api")
        assert limiter1 is limiter2

    def test_default_limits(self):
        """Test that default limits are defined for known APIs."""
        assert "datos.gob.es" in DEFAULT_LIMITS
        assert "ine" in DEFAULT_LIMITS
        assert "aemet" in DEFAULT_LIMITS
        assert "boe" in DEFAULT_LIMITS

    @pytest.mark.asyncio
    async def test_acquire_single(self):
        """Test acquiring a single rate limit token."""
        await RateLimiter.acquire("test_api")
        # Should complete without error

    @pytest.mark.asyncio
    async def test_acquire_multiple_within_limit(self):
        """Test acquiring multiple tokens within rate limit."""
        # Default limit is 10/second for unknown APIs
        for _ in range(5):
            await RateLimiter.acquire("test_api")
        # Should complete without significant delay

    def test_reset_single(self):
        """Test resetting a single limiter."""
        RateLimiter.get_limiter("api1")
        RateLimiter.get_limiter("api2")
        RateLimiter.reset("api1")
        assert "api1" not in RateLimiter._limiters
        assert "api2" in RateLimiter._limiters

    def test_reset_all(self):
        """Test resetting all limiters."""
        RateLimiter.get_limiter("api1")
        RateLimiter.get_limiter("api2")
        RateLimiter.reset()
        assert len(RateLimiter._limiters) == 0

    def test_get_stats(self):
        """Test getting rate limiter statistics."""
        RateLimiter.get_limiter("datos.gob.es")
        stats = RateLimiter.get_stats()
        assert "datos.gob.es" in stats
        assert stats["datos.gob.es"]["active"] is True
        assert stats["datos.gob.es"]["rate_per_second"] == 10.0

    def test_custom_rate_from_env(self):
        """Test that rate limits can be configured from environment."""
        with patch.dict("os.environ", {"RATE_LIMIT_TEST_API": "20"}):
            rate = RateLimiter._get_rate_limit("test_api")
            assert rate == 20.0


# =============================================================================
# HTTP Client Tests
# =============================================================================


class TestHTTPClient:
    """Tests for the HTTP client."""

    @pytest.fixture
    def client(self):
        """Create a test HTTP client."""
        return HTTPClient("test_api", "https://api.example.com/", rate_limit=False)

    @pytest.mark.asyncio
    async def test_get_request(self, client):
        """Test basic GET request."""
        with respx.mock:
            respx.get("https://api.example.com/endpoint").respond(
                status_code=200,
                json={"result": "success"}
            )

            response = await client.get("endpoint")
            assert response.status_code == 200
            assert response.json() == {"result": "success"}

    @pytest.mark.asyncio
    async def test_get_json(self, client):
        """Test GET request with JSON response."""
        with respx.mock:
            respx.get("https://api.example.com/data").respond(
                status_code=200,
                json={"data": [1, 2, 3]}
            )

            data = await client.get_json("data")
            assert data == {"data": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_get_with_params(self, client):
        """Test GET request with query parameters."""
        with respx.mock:
            route = respx.get("https://api.example.com/search").respond(
                status_code=200,
                json={"results": []}
            )

            await client.get("search", params={"q": "test", "page": 1})
            assert route.called
            assert route.calls[0].request.url.params["q"] == "test"

    @pytest.mark.asyncio
    async def test_post_request(self, client):
        """Test POST request."""
        with respx.mock:
            respx.post("https://api.example.com/create").respond(
                status_code=201,
                json={"id": 123}
            )

            response = await client.post("create", json_data={"name": "test"})
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_http_error(self, client):
        """Test HTTP error handling."""
        with respx.mock:
            respx.get("https://api.example.com/notfound").respond(
                status_code=404,
                json={"error": "Not found"}
            )

            with pytest.raises(HTTPClientError) as exc_info:
                await client.get("notfound")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_timeout_error(self, client):
        """Test timeout handling."""
        import httpx

        with respx.mock:
            respx.get("https://api.example.com/slow").mock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            with pytest.raises(HTTPClientError) as exc_info:
                await client.get("slow")

            assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_request_error(self, client):
        """Test request error handling."""
        import httpx

        with respx.mock:
            respx.get("https://api.example.com/error").mock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(HTTPClientError) as exc_info:
                await client.get("error")

            assert "failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_absolute_url(self, client):
        """Test request with absolute URL."""
        with respx.mock:
            respx.get("https://other.example.com/external").respond(
                status_code=200,
                json={"external": True}
            )

            response = await client.get("https://other.example.com/external")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_custom_headers(self, client):
        """Test request with custom headers."""
        with respx.mock:
            route = respx.get("https://api.example.com/auth").respond(
                status_code=200,
                json={"authenticated": True}
            )

            await client.get("auth", headers={"Authorization": "Bearer token"})
            assert route.called
            assert route.calls[0].request.headers["Authorization"] == "Bearer token"

    @pytest.mark.asyncio
    async def test_no_raise_for_status(self, client):
        """Test disabling raise_for_status."""
        with respx.mock:
            respx.get("https://api.example.com/error").respond(
                status_code=500,
                json={"error": "Internal error"}
            )

            # Should not raise with raise_for_status=False
            response = await client.get("error", raise_for_status=False)
            assert response.status_code == 500


class TestHTTPClientWithRateLimiting:
    """Tests for HTTP client with rate limiting enabled."""

    def setup_method(self):
        """Reset rate limiters before each test."""
        RateLimiter.reset()

    @pytest.mark.asyncio
    async def test_rate_limiting_applied(self):
        """Test that rate limiting is applied to requests."""
        client = HTTPClient("test_api", "https://api.example.com/", rate_limit=True)

        with respx.mock:
            respx.get("https://api.example.com/data").respond(
                status_code=200,
                json={"ok": True}
            )

            # Make request
            await client.get("data")

            # Rate limiter should have been created
            assert "test_api" in RateLimiter._limiters

    @pytest.mark.asyncio
    async def test_rate_limiting_disabled(self):
        """Test that rate limiting can be disabled."""
        client = HTTPClient("no_limit_api", "https://api.example.com/", rate_limit=False)

        with respx.mock:
            respx.get("https://api.example.com/data").respond(
                status_code=200,
                json={"ok": True}
            )

            await client.get("data")

            # Rate limiter should not have been created
            assert "no_limit_api" not in RateLimiter._limiters
