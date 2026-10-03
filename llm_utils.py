"""
llm_utils.py
One place for LLM-call error handling, so every call site behaves the same way.

Transient failures (rate limits, timeouts, connection drops, 5xx) are retried
with exponential backoff. Permanent failures (bad key, unknown model, bad
request) fail immediately, because retrying them only wastes time.
"""

import time

# HTTP status codes where trying again cannot help.
# 429 is deliberately absent: rate limits are the single most common transient
# failure for hosted model providers and are exactly what backoff is for.
NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails for good (retries exhausted or non-retryable).

    `status` is the provider's HTTP status when there was one, so callers can
    react to *why* it failed instead of parsing the message. That distinction
    matters for structured-output fallback: a 400 means this decode strategy was
    rejected and another one is worth trying, while a 401 means stop.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _status_of(err: BaseException) -> int | None:
    """Best-effort HTTP status from an SDK exception.

    Providers disagree on where they put it: some set `.status_code`, some only
    populate an httpx `.response`. Checking only one of those made a 401 look
    like an unknown error and got it needlessly retried three times.
    """
    status = getattr(err, "status_code", None)
    if status is None:
        status = getattr(getattr(err, "response", None), "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status


def safe_invoke(llm, messages, retries: int = 3, base_delay: float = 1.0, sleep=time.sleep):
    """Call llm.invoke(messages), retrying transient failures with backoff.

    `sleep` is injectable so tests can run instantly and assert on the delays.
    """
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as err:  # noqa: BLE001 - SDKs raise many exception types
            last_error = err
            status = _status_of(err)
            if status in NON_RETRYABLE_STATUS:
                raise LLMCallError(
                    f"LLM call failed with non-retryable status {status}: {err}",
                    status=status,
                ) from err
            if attempt < retries:
                sleep(base_delay * (2 ** (attempt - 1)))  # 1s, 2s, 4s...
    raise LLMCallError(
        f"LLM call failed after {retries} attempts: {last_error}",
        status=_status_of(last_error) if last_error else None,
    ) from last_error
