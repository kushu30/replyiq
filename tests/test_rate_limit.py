from unittest.mock import MagicMock

import pytest
from groq import RateLimitError

from generator.rate_limit import retry_with_backoff


def _make_rate_limit_error():
    fake_response = MagicMock()
    fake_response.status_code = 429
    fake_response.headers = {}
    fake_response.request = MagicMock()
    return RateLimitError("rate limited", response=fake_response, body=None)


def test_retry_succeeds_after_transient_failure(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    call_count = {"count": 0}

    @retry_with_backoff(max_retries=3, base_delay=1)
    def flaky_function():
        call_count["count"] += 1
        if call_count["count"] < 2:
            raise _make_rate_limit_error()
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count["count"] == 2


def test_retry_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    @retry_with_backoff(max_retries=2, base_delay=1)
    def always_fails():
        raise _make_rate_limit_error()

    with pytest.raises(RateLimitError):
        always_fails()


def test_retry_does_not_interfere_with_successful_call():
    @retry_with_backoff()
    def always_succeeds():
        return "ok"

    assert always_succeeds() == "ok"