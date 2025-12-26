"""Rate limiting for API requests using aiolimiter."""

import os
from typing import ClassVar

from aiolimiter import AsyncLimiter

from .logging import get_logger

logger = get_logger("ratelimit")


# Default rate limits (requests per second)
DEFAULT_LIMITS: dict[str, float] = {
    "datos.gob.es": 10.0,  # Main API - 10 req/s
    "ine": 5.0,            # INE API - 5 req/s (conservative)
    "aemet": 10.0,         # AEMET API - 10 req/s
    "boe": 10.0,           # BOE API - 10 req/s
}


class RateLimiter:
    """Async rate limiter for multiple APIs.

    Uses token bucket algorithm to limit requests per second for each API.
    Rate limits can be configured via environment variables:
    - RATE_LIMIT_DATOS_GOB (default: 10)
    - RATE_LIMIT_INE (default: 5)
    - RATE_LIMIT_AEMET (default: 10)
    - RATE_LIMIT_BOE (default: 10)
    """

    _limiters: ClassVar[dict[str, AsyncLimiter]] = {}

    @classmethod
    def _get_rate_limit(cls, api_name: str) -> float:
        """Get rate limit for an API from env var or default."""
        env_var_name = f"RATE_LIMIT_{api_name.upper().replace('.', '_').replace('-', '_')}"
        env_value = os.getenv(env_var_name)

        if env_value:
            try:
                return float(env_value)
            except ValueError:
                logger.warning(
                    "invalid_rate_limit",
                    api=api_name,
                    env_var=env_var_name,
                    value=env_value,
                )

        return DEFAULT_LIMITS.get(api_name, 10.0)

    @classmethod
    def get_limiter(cls, api_name: str) -> AsyncLimiter:
        """Get or create a rate limiter for the specified API.

        Args:
            api_name: Name of the API (e.g., 'datos.gob.es', 'ine', 'aemet', 'boe')

        Returns:
            AsyncLimiter instance for the API.
        """
        if api_name not in cls._limiters:
            rate = cls._get_rate_limit(api_name)
            # AsyncLimiter(max_rate, time_period)
            # e.g., AsyncLimiter(10, 1) = 10 requests per 1 second
            cls._limiters[api_name] = AsyncLimiter(rate, 1.0)
            logger.debug(
                "rate_limiter_created",
                api=api_name,
                rate_per_second=rate,
            )
        return cls._limiters[api_name]

    @classmethod
    async def acquire(cls, api_name: str) -> None:
        """Acquire a rate limit token for the specified API.

        This method will wait if the rate limit has been reached.

        Args:
            api_name: Name of the API to acquire a token for.
        """
        limiter = cls.get_limiter(api_name)
        await limiter.acquire()

    @classmethod
    def reset(cls, api_name: str | None = None) -> None:
        """Reset rate limiter(s).

        Args:
            api_name: If provided, reset only this API's limiter.
                     If None, reset all limiters.
        """
        if api_name:
            if api_name in cls._limiters:
                del cls._limiters[api_name]
                logger.debug("rate_limiter_reset", api=api_name)
        else:
            cls._limiters.clear()
            logger.debug("rate_limiters_reset_all")

    @classmethod
    def get_stats(cls) -> dict[str, dict]:
        """Get statistics about rate limiters.

        Returns:
            Dict with API names and their rate limit configuration.
        """
        return {
            api: {
                "rate_per_second": cls._get_rate_limit(api),
                "active": api in cls._limiters,
            }
            for api in DEFAULT_LIMITS
        }
