"""Tests for retry utilities."""

from __future__ import annotations

import time

import pytest

from nhl_pipeline.utils import RetryError, RateLimiter, with_retry


class TestRetryDecorator:
    def test_succeeds_on_first_try(self):
        call_count = [0]

        @with_retry(max_retries=3)
        def fn():
            call_count[0] += 1
            return "ok"

        result = fn()
        assert result == "ok"
        assert call_count[0] == 1

    def test_retries_on_failure_then_succeeds(self):
        call_count = [0]

        @with_retry(max_retries=3, backoff_base=0.01)
        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("transient")
            return "ok"

        result = fn()
        assert result == "ok"
        assert call_count[0] == 3

    def test_raises_retry_error_after_exhaustion(self):
        @with_retry(max_retries=2, backoff_base=0.01)
        def fn():
            raise RuntimeError("always fails")

        with pytest.raises(RetryError):
            fn()

    def test_only_retries_specified_exceptions(self):
        call_count = [0]

        @with_retry(max_retries=3, backoff_base=0.01, exceptions=(ValueError,))
        def fn():
            call_count[0] += 1
            raise TypeError("wrong type — should not retry")

        with pytest.raises(TypeError):
            fn()

        assert call_count[0] == 1  # did not retry

    def test_retry_count(self):
        call_count = [0]

        @with_retry(max_retries=2, backoff_base=0.01)
        def fn():
            call_count[0] += 1
            raise RuntimeError("fail")

        with pytest.raises(RetryError):
            fn()

        assert call_count[0] == 3  # 1 initial + 2 retries


class TestRateLimiter:
    def test_wait_enforces_min_delay(self):
        limiter = RateLimiter(min_delay=0.05)
        limiter.wait()  # first call — no delay needed
        t0 = time.monotonic()
        limiter.wait()  # second call — should wait ~0.05s
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.04  # allow a small tolerance

    def test_no_extra_wait_after_sufficient_gap(self):
        limiter = RateLimiter(min_delay=0.05)
        limiter.wait()
        time.sleep(0.1)  # more than min_delay
        t0 = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.02  # should return quickly
