"""Shared utilities: logging, retry, rate-limiting."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger("nhl_pipeline.utils")


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""


def with_retry(
    max_retries: int = 4,
    backoff_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    logger: logging.Logger | None = None,
) -> Callable[[F], F]:
    """Decorator: retry *func* up to *max_retries* times with exponential backoff.

    Args:
        max_retries:   Number of retry attempts after the first failure.
        backoff_base:  Base delay in seconds; delay = backoff_base ** attempt.
        exceptions:    Exception types that trigger a retry.
        logger:        Logger to use for warnings; defaults to module logger.
    """
    _log = logger or log

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    delay = backoff_base ** (attempt + 1)
                    _log.warning(
                        "Attempt %d/%d failed for %s — retrying in %.1fs. Error: %s",
                        attempt + 1, max_retries + 1, func.__qualname__, delay, exc,
                    )
                    time.sleep(delay)
            raise RetryError(
                f"All {max_retries + 1} attempts failed for {func.__qualname__}"
            ) from last_exc

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Rate-limiting helper
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket–style rate limiter (min delay between calls)."""

    def __init__(self, min_delay: float = 0.5) -> None:
        self._min_delay = min_delay
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_call = time.monotonic()
