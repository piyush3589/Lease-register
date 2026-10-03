"""
Tests for the structured-output fallback chain.

These exist because of a real failure: Groq accepted the request for strict
Structured Outputs and then rejected it at generation time with 400
`json_validate_failed` (openai/gpt-oss models return an empty completion when
constrained decoding is forced on them). The old code picked the first strategy
that could be *built*, so that one rejection failed the whole extraction even
though json_mode would have worked.

Offline: the fake below reproduces the provider's behaviour, no key or network.
"""

import pytest

import extractor
from extractor import ExtractionResult, extract_lease_details
from llm_utils import LLMCallError

DOCUMENT = (
    "This Lease Agreement is made between Mr. Arvind Shah (Landlord) and "
    "Ms. Priya Nair (Tenant). Monthly rent Rs. 65,000."
)

GOOD_RESULT = ExtractionResult(
    fields=[
        {"field": "tenant_name", "value": "Priya Nair", "source_quote": "Ms. Priya Nair (Tenant)"},
        {"field": "monthly_rent", "value": "Rs. 65,000", "source_quote": "Monthly rent Rs. 65,000."},
    ]
)


class FakeStructuredFailure(Exception):
    """Stands in for an SDK exception carrying a status code."""

    def __init__(self, status):
        super().__init__(f"provider said {status}")
        self.status_code = status


class FakeLLM:
    """Rejects the first `failing` strategies, then answers.

    `rejected` is the status the provider returns for those strategies, which is
    what decides whether a fallback is attempted at all.
    """

    def __init__(self, rejected=400, failing=1, unsupported=(), result=GOOD_RESULT):
        self.rejected = rejected
        self.failing = failing
        self.unsupported = set(unsupported)
        self.result = result
        self.attempts = []
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        key = _strategy_name(kwargs)
        if key in self.unsupported:
            raise TypeError(f"{key} not supported")
        self.attempts.append(key)
        return self

    def invoke(self, messages):
        # Counts invocations, not strategies: `failing=1` means the very first
        # call is refused and the next one succeeds, whatever strategy that is.
        self.calls += 1
        if self.calls <= self.failing:
            raise FakeStructuredFailure(self.rejected)
        return self.result


def _strategy_name(kwargs) -> str:
    if kwargs.get("method") == "json_schema":
        return "strict" if kwargs.get("strict") else "json_schema"
    return kwargs.get("method", "unknown")


# --- the chain is walked at invoke time ------------------------------------


def test_strict_is_tried_first():
    llm = FakeLLM(failing=0)
    extract_lease_details(llm, DOCUMENT)
    assert llm.attempts[0] == "strict"


def test_a_rejected_strict_strategy_falls_back():
    """The bug this file is about."""
    llm = FakeLLM(rejected=400, failing=1)
    result = extract_lease_details(llm, DOCUMENT)
    assert llm.attempts[0] == "strict"
    assert llm.attempts[1] == "json_schema"
    assert result.details.tenant_name == "Priya Nair"


def test_falling_back_marks_the_reply_degraded():
    """A non-strict reply is trusted less, and the caller must be told."""
    assert extract_lease_details(FakeLLM(failing=1), DOCUMENT).degraded is True


def test_a_working_strict_call_is_not_degraded():
    assert extract_lease_details(FakeLLM(failing=0), DOCUMENT).degraded is False


def test_the_chain_keeps_going_past_json_schema():
    llm = FakeLLM(rejected=400, failing=2)
    result = extract_lease_details(llm, DOCUMENT)
    assert llm.attempts == ["strict", "json_schema", "json_mode"]
    assert result.degraded is True


def test_an_unsupported_strategy_is_skipped_without_calling_it():
    """A provider with no strict support should not even try strict."""
    llm = FakeLLM(unsupported=("strict", "json_schema"), failing=0)
    result = extract_lease_details(llm, DOCUMENT)
    assert llm.attempts == ["json_mode"]
    assert result.degraded is True


def test_every_strategy_is_tried_before_giving_up():
    llm = FakeLLM(rejected=400, failing=99)
    with pytest.raises(LLMCallError):
        extract_lease_details(llm, DOCUMENT)
    assert llm.attempts == ["strict", "json_schema", "json_mode", "function_calling"]


# --- which failures justify a fallback --------------------------------------


def test_a_bad_key_does_not_trigger_a_fallback():
    """401 will fail every strategy identically, so retrying wastes a call."""
    llm = FakeLLM(rejected=401, failing=99)
    with pytest.raises(LLMCallError):
        extract_lease_details(llm, DOCUMENT)
    assert llm.attempts == ["strict"]


def test_a_forbidden_key_does_not_trigger_a_fallback():
    llm = FakeLLM(rejected=403, failing=99)
    with pytest.raises(LLMCallError):
        extract_lease_details(llm, DOCUMENT)
    assert llm.attempts == ["strict"]


def test_an_unknown_model_does_not_trigger_a_fallback():
    llm = FakeLLM(rejected=404, failing=99)
    with pytest.raises(LLMCallError):
        extract_lease_details(llm, DOCUMENT)
    # 404 means the model is wrong, not the strategy, so it is not retried
    # down the chain; the error surfaces immediately.
    assert llm.attempts == ["strict"]


def test_a_schema_rejection_falls_back():
    llm = FakeLLM(rejected=422, failing=1)
    extract_lease_details(llm, DOCUMENT)
    assert llm.attempts[1] == "json_schema"


# --- prose last resort -----------------------------------------------------


class ProseOnlyLLM:
    """Every structured strategy is refused; plain text still works."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= 4:
            raise FakeStructuredFailure(400)
        return type("R", (), {"content": '```json\n{"fields": [{"field": "tenant_name", '
                          '"value": "Priya Nair", "source_quote": "Ms. Priya Nair (Tenant)"}]}\n```'})()


def test_it_still_extracts_when_only_prose_is_left():
    llm = ProseOnlyLLM()
    result = extract_lease_details(llm, DOCUMENT)
    assert result.details.tenant_name == "Priya Nair"
    assert result.degraded is True


# --- the strategy list is the documented one -------------------------------


def test_attempts_are_ordered_strict_first():
    keys = [_strategy_name(kwargs) for kwargs, _ in extractor._STRUCTURE_ATTEMPTS]
    assert keys == ["strict", "json_schema", "json_mode", "function_calling"]


def test_only_the_strict_attempt_is_not_degraded():
    flags = [flag for _, flag in extractor._STRUCTURE_ATTEMPTS]
    assert flags == [False, True, True, True]


def test_a_rejected_status_list_excludes_auth_and_missing_model():
    """400/422 mean "try another strategy"; 401/403/404 would all fail anyway."""
    assert extractor._STRATEGY_REJECTED_STATUS == {400, 422}
    assert 401 not in extractor._STRATEGY_REJECTED_STATUS
    assert 403 not in extractor._STRATEGY_REJECTED_STATUS
    assert 404 not in extractor._STRATEGY_REJECTED_STATUS