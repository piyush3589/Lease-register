"""
Contract tests for the request we send to Groq.

These need no API key and make no network call: they inspect the request
langchain-groq builds. The whole design rests on the default model supporting
strict Structured Outputs, and on strict mode still allowing nulls -- if either
stops being true the "missing fields come back null" promise breaks silently.

A langchain upgrade that changes this schema shape should fail here.
"""

import pytest

from extractor import FIELD_NAMES, ExtractionResult, _structured_runner

langchain_groq = pytest.importorskip("langchain_groq")

# Imported, not repeated, so this contract test cannot drift from the model the
# service actually sends.
from api import DEFAULT_MODEL


def build_llm(model=DEFAULT_MODEL):
    return langchain_groq.ChatGroq(
        model=model,
        temperature=0,
        api_key="fake-key-schema-introspection-only",
        max_retries=0,
    )


def request_kwargs(runnable):
    """with_structured_output returns `llm.bind(...) | parser`, so the request
    kwargs live on the first step of the sequence."""
    target = getattr(runnable, "first", runnable)
    return getattr(target, "kwargs", {})


@pytest.fixture(scope="module")
def schema():
    runner = build_llm().with_structured_output(
        ExtractionResult, method="json_schema", strict=True
    )
    fmt = request_kwargs(runner)["response_format"]
    assert fmt["type"] == "json_schema"
    return fmt["json_schema"]


def test_strict_mode_is_requested(schema):
    assert schema["strict"] is True


def test_schema_is_named_after_the_model(schema):
    assert schema["name"] == "ExtractionResult"


def test_schema_is_json_serialisable(schema):
    import json

    assert json.loads(json.dumps(schema["schema"]))


def test_top_level_forbids_extra_keys(schema):
    assert schema["schema"]["additionalProperties"] is False


def test_every_row_field_is_required(schema):
    """Strict mode requires this; without it the model may omit keys."""
    item = schema["schema"]["properties"]["fields"]["items"]
    assert set(item["required"]) == {"field", "value", "source_quote"}
    assert item["additionalProperties"] is False


def test_field_column_is_an_enum_of_exactly_the_nine_fields(schema):
    """Constrained decoding can only guarantee a valid shape if the set of
    legal field names is finite and declared."""
    item = schema["schema"]["properties"]["fields"]["items"]
    assert item["properties"]["field"]["enum"] == list(FIELD_NAMES)
    assert len(FIELD_NAMES) == 9


def test_value_and_quote_are_nullable(schema):
    """The critical property: strict decoding must still permit null, or the
    'missing fields come back as null' promise becomes 'model must invent'."""
    item = schema["schema"]["properties"]["fields"]["items"]
    for column in ("value", "source_quote"):
        variants = item["properties"][column]["anyOf"]
        assert {"type": "null"} in variants, f"{column} must allow null"
        assert {"type": "string"} in variants


def test_json_mode_is_the_first_fallback():
    runner = build_llm().with_structured_output(ExtractionResult, method="json_mode")
    assert request_kwargs(runner)["response_format"] == {"type": "json_object"}


def test_function_calling_is_the_last_resort():
    runner = build_llm().with_structured_output(
        ExtractionResult, method="function_calling"
    )
    tools = request_kwargs(runner)["tools"]
    assert tools[0]["function"]["name"] == "ExtractionResult"


def test_structured_runner_picks_strict_first():
    """The probe must land on strict decoding, since that is the whole point."""
    runner = _structured_runner(build_llm())
    assert request_kwargs(runner)["response_format"]["json_schema"]["strict"] is True


def test_structured_runner_falls_back_when_method_is_unsupported():
    class NoStructuredOutput:
        def with_structured_output(self, schema, **kwargs):
            if kwargs.get("method") == "json_schema":
                raise ValueError("json_schema is not supported for this model")
            raise TypeError("nope")

        def invoke(self, messages):
            return "unused"

    assert _structured_runner(NoStructuredOutput()) is None


def test_structured_runner_returns_none_without_the_method():
    class Bare:
        pass

    assert _structured_runner(Bare()) is None


def test_degraded_flag_is_set_only_when_constrained_decoding_is_unavailable():
    """A silent downgrade to plain JSON must be visible in the response."""

    class PlainOnly:
        def with_structured_output(self, schema, **kwargs):
            raise TypeError("no structured output here")

        def invoke(self, messages):
            from types import SimpleNamespace

            return SimpleNamespace(content='{"tenant_name": "A"}')

    from extractor import extract_lease_details

    result = extract_lease_details(PlainOnly(), "TENANT: A")
    assert result.degraded is True
    assert result.details.tenant_name == "A"
