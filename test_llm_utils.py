"""
Tests for the retry layer. This module had no coverage in v1 even though it is
the piece most likely to fail subtly: a wrong backoff or a retried 401 is
invisible until production.

`sleep` is injected, so these run instantly and can assert on the delays.
"""

import pytest

from llm_utils import LLMCallError, NON_RETRYABLE_STATUS, safe_invoke


class FakeError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class HttpXStyleError(Exception):
    """Providers that only populate `.response.status_code`, like httpx errors."""

    def __init__(self, message, status):
        super().__init__(message)
        self.response = type("Response", (), {"status_code": status})()


class FlakyLLM:
    """Fails `failures` times with `status`, then returns a result."""

    def __init__(self, failures, status=None, result="ok"):
        self.failures = failures
        self.status = status
        self.result = result
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise FakeError(f"boom #{self.calls}", self.status)
        return self.result


class Recorder:
    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


def test_succeeds_on_first_attempt_without_sleeping():
    llm = FlakyLLM(failures=0)
    sleep = Recorder()
    assert safe_invoke(llm, ["m"], sleep=sleep) == "ok"
    assert llm.calls == 1
    assert sleep.delays == []


def test_retries_transient_failure_then_succeeds():
    llm = FlakyLLM(failures=2)
    sleep = Recorder()
    assert safe_invoke(llm, ["m"], sleep=sleep) == "ok"
    assert llm.calls == 3


def test_backoff_is_exponential():
    # 4 attempts => 3 sleeps. The final attempt is never followed by a sleep,
    # so there is no point waiting before a failure we are about to raise.
    llm = FlakyLLM(failures=99, status=500)
    sleep = Recorder()
    with pytest.raises(LLMCallError):
        safe_invoke(llm, ["m"], retries=4, base_delay=1.0, sleep=sleep)
    assert sleep.delays == [1.0, 2.0, 4.0]


def test_base_delay_scales_the_backoff():
    llm = FlakyLLM(failures=99, status=503)
    sleep = Recorder()
    with pytest.raises(LLMCallError):
        safe_invoke(llm, ["m"], base_delay=0.5, sleep=sleep)
    assert sleep.delays == [0.5, 1.0]


def test_raises_after_retries_exhausted():
    llm = FlakyLLM(failures=99, status=500)
    sleep = Recorder()
    with pytest.raises(LLMCallError, match="after 3 attempts"):
        safe_invoke(llm, ["m"], sleep=sleep)
    assert llm.calls == 3
    assert len(sleep.delays) == 2  # no sleep after the final attempt


@pytest.mark.parametrize("status", sorted(NON_RETRYABLE_STATUS))
def test_non_retryable_status_fails_immediately(status):
    llm = FlakyLLM(failures=99, status=status)
    sleep = Recorder()
    with pytest.raises(LLMCallError, match="non-retryable"):
        safe_invoke(llm, ["m"], sleep=sleep)
    assert llm.calls == 1, "a permanent failure must not be retried"
    assert sleep.delays == []


def test_rate_limit_is_retried_because_429_is_transient():
    assert 429 not in NON_RETRYABLE_STATUS
    llm = FlakyLLM(failures=1, status=429)
    sleep = Recorder()
    assert safe_invoke(llm, ["m"], sleep=sleep) == "ok"
    assert llm.calls == 2


def test_status_on_nested_response_is_detected():
    """httpx-style errors hide the status one level down; a 401 there must not
    be retried three times."""

    class NestedLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise HttpXStyleError("unauthorized", 401)

    llm = NestedLLM()
    with pytest.raises(LLMCallError, match="non-retryable status 401"):
        safe_invoke(llm, ["m"], sleep=Recorder())
    assert llm.calls == 1


def test_error_without_status_is_treated_as_transient():
    llm = FlakyLLM(failures=99, status=None)
    with pytest.raises(LLMCallError, match="after 3 attempts"):
        safe_invoke(llm, ["m"], sleep=Recorder())
    assert llm.calls == 3


def test_non_integer_status_is_ignored():
    class WeirdLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise FakeError("odd", status_code="401")

    llm = WeirdLLM()
    with pytest.raises(LLMCallError, match="after 3 attempts"):
        safe_invoke(llm, ["m"], sleep=Recorder())
    assert llm.calls == 3, "a string status must not match the int set by accident"


def test_boolean_status_is_not_treated_as_int():
    class BoolLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise FakeError("odd", status_code=True)

    llm = BoolLLM()
    with pytest.raises(LLMCallError, match="after 3 attempts"):
        safe_invoke(llm, ["m"], sleep=Recorder())
    assert llm.calls == 3


def test_single_attempt_mode_never_sleeps():
    llm = FlakyLLM(failures=99, status=500)
    sleep = Recorder()
    with pytest.raises(LLMCallError, match="after 1 attempts"):
        safe_invoke(llm, ["m"], retries=1, sleep=sleep)
    assert llm.calls == 1
    assert sleep.delays == []
