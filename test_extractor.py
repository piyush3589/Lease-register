"""
Tests for extraction: schema handling, provenance, and the no-guessing promise.

No API key needed. Fakes stand in for Groq at two levels:

  StrictLLM   -- supports with_structured_output, as the real ChatGroq does.
  PlainLLM    -- does not, forcing the raw-invoke + JSON-salvage path.

Run with: pytest -v
"""

import json
from types import SimpleNamespace

import pytest

from extractor import (
    FIELD_NAMES,
    ExtractionResult,
    ExtractedField,
    LeaseDetails,
    extract_lease_details,
)

RESIDENTIAL = """RESIDENTIAL LEASE AGREEMENT

This Agreement is made on 1st April 2025.

LANDLORD: Arvind Shah
TENANT: Priya Nair

PREMISES: Flat No. 1204, Oberoi Splendor, Andheri East, Mumbai 400053

TERM: The lease commences on 1st April 2025 and expires on 28th February 2026,
a term of eleven (11) months.

RENT: The Tenant shall pay a monthly rent of Rs. 65,000 (Sixty Five Thousand
Rupees only).

SECURITY DEPOSIT: An interest-free refundable deposit of Rs. 3,25,000 has been
paid by the Tenant.

RENEWAL: The lease is renewable on mutual consent with rent revision up to 10%.
"""

INCOMPLETE = """LEASE NOTE (Informal)

Tenant: Rohan Mehta
Property: 2BHK, Hiranandani Gardens, Powai, Mumbai

Rent of Rs. 48,000 per month agreed starting this month. Deposit to be discussed
separately.
"""


def rows(**overrides):
    """Build a full nine-row result, applying overrides per field."""
    values = {
        "tenant_name": ("Priya Nair", "TENANT: Priya Nair"),
        "landlord_name": ("Arvind Shah", "LANDLORD: Arvind Shah"),
        "property_address": (
            "Flat No. 1204, Oberoi Splendor, Andheri East, Mumbai 400053",
            "PREMISES: Flat No. 1204, Oberoi Splendor, Andheri East, Mumbai 400053",
        ),
        "lease_start_date": ("1st April 2025", "The lease commences on 1st April 2025"),
        "lease_end_date": ("28th February 2026", "expires on 28th February 2026"),
        "lease_term": ("eleven (11) months", "a term of eleven (11) months"),
        "monthly_rent": ("Rs. 65,000", "monthly rent of Rs. 65,000"),
        "security_deposit": ("Rs. 3,25,000", "deposit of Rs. 3,25,000 has been"),
        "renewal_terms": (
            "renewable on mutual consent with rent revision up to 10%",
            "renewable on mutual consent with rent revision up to 10%",
        ),
    }
    for key in list(values):
        if key in overrides:
            values[key] = overrides[key]
    return ExtractionResult(
        fields=[ExtractedField(field=k, value=v, source_quote=q) for k, (v, q) in values.items()]
    )


class StrictLLM:
    """Mimics ChatGroq: supports with_structured_output and returns parsed rows."""

    def __init__(self, result, fail_first=0):
        self.result = result
        self.fail_first = fail_first
        self.calls = 0
        self.prompts = []

    def with_structured_output(self, schema, **kwargs):
        self.structured_kwargs = kwargs
        outer = self

        class Runner:
            def invoke(self, messages):
                outer.calls += 1
                outer.prompts = [m.content for m in messages]
                if outer.calls <= outer.fail_first:
                    raise RuntimeError("transient")
                return outer.result

        return Runner()


class PlainLLM:
    """No with_structured_output: forces the raw-invoke + salvage path."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.reply)


def flat_json(**overrides):
    data = {
        "tenant_name": "Priya Nair",
        "landlord_name": "Arvind Shah",
        "property_address": "Flat No. 1204, Oberoi Splendor, Andheri East, Mumbai 400053",
        "lease_start_date": "1st April 2025",
        "lease_end_date": "28th February 2026",
        "lease_term": "eleven (11) months",
        "monthly_rent": "Rs. 65,000",
        "security_deposit": "Rs. 3,25,000",
        "renewal_terms": "renewable on mutual consent",
    }
    data.update(overrides)
    return json.dumps(data)


# --- structured path -------------------------------------------------------


def test_extracts_all_fields_from_complete_document():
    result = extract_lease_details(StrictLLM(rows()), RESIDENTIAL)
    assert result.details.tenant_name == "Priya Nair"
    assert result.details.monthly_rent == "Rs. 65,000"
    assert result.details.landlord_name == "Arvind Shah"
    assert result.missing == []
    assert result.no_fields_found is False


def test_prefers_strict_constrained_decoding():
    llm = StrictLLM(rows())
    extract_lease_details(llm, RESIDENTIAL)
    assert llm.structured_kwargs == {"method": "json_schema", "strict": True}


def test_missing_fields_come_back_as_none_not_guessed():
    sparse = rows(
        landlord_name=(None, None),
        lease_start_date=(None, None),
        lease_end_date=(None, None),
        lease_term=(None, None),
        security_deposit=(None, None),
        renewal_terms=(None, None),
        tenant_name=("Rohan Mehta", "Tenant: Rohan Mehta"),
        property_address=(
            "2BHK, Hiranandani Gardens, Powai, Mumbai",
            "Property: 2BHK, Hiranandani Gardens, Powai, Mumbai",
        ),
        monthly_rent=("Rs. 48,000", "Rent of Rs. 48,000 per month"),
    )
    result = extract_lease_details(StrictLLM(sparse), INCOMPLETE)
    assert result.details.tenant_name == "Rohan Mehta"
    assert result.details.monthly_rent == "Rs. 48,000"
    assert result.details.landlord_name is None
    assert result.details.security_deposit is None
    assert result.details.renewal_terms is None
    assert "lease_term" in result.missing


def test_evidence_is_returned_for_each_field():
    result = extract_lease_details(StrictLLM(rows()), RESIDENTIAL)
    assert result.evidence["monthly_rent"] == "monthly rent of Rs. 65,000"
    assert len(result.evidence) == len(FIELD_NAMES)


def test_verified_extraction_has_no_unverified_fields():
    result = extract_lease_details(StrictLLM(rows()), RESIDENTIAL)
    assert result.unverified_fields == []


def test_quote_absent_from_document_is_flagged_unverified():
    """A value with a quote the document does not contain is a fabrication
    risk, and must surface rather than pass as verified."""
    tampered = rows(monthly_rent=("Rs. 1", "monthly rent of Rs. 1"))
    result = extract_lease_details(StrictLLM(tampered), RESIDENTIAL)
    assert result.details.monthly_rent == "Rs. 1"
    assert result.unverified_fields == ["monthly_rent"]


def test_value_absent_from_its_own_quote_is_flagged_unverified():
    """Quote is real, but does not contain the value: stitched from elsewhere."""
    tampered = rows(monthly_rent=("Rs. 99,999", "monthly rent of Rs. 65,000"))
    result = extract_lease_details(StrictLLM(tampered), RESIDENTIAL)
    assert result.unverified_fields == ["monthly_rent"]


def test_null_value_is_never_flagged_unverified():
    """A missing field is not an error, so it must not pollute the review list."""
    result = extract_lease_details(
        StrictLLM(rows(renewal_terms=(None, None))), RESIDENTIAL
    )
    assert result.details.renewal_terms is None
    assert result.unverified_fields == []


def test_missing_quote_is_flagged_unverified():
    result = extract_lease_details(
        StrictLLM(rows(lease_term=("eleven months", None))), RESIDENTIAL
    )
    assert result.unverified_fields == ["lease_term"]


def test_quote_matching_ignores_case_and_punctuation():
    """PDF extraction mangles spacing and dashes; a cosmetic difference must not
    make a real quote look fabricated."""
    result = extract_lease_details(
        StrictLLM(
            rows(
                monthly_rent=(
                    "Rs. 65,000",
                    "monthly  rent   of Rs. 65,000",
                )
            )
        ),
        RESIDENTIAL,
    )
    assert result.unverified_fields == []


def test_unexpected_keys_are_surfaced_not_dropped_silently():
    class DictsLLM(StrictLLM):
        def with_structured_output(self, schema, **kwargs):
            outer = self

            class Runner:
                def invoke(self, messages):
                    return {"fields": rows().model_dump()["fields"], "sqft": "1200"}

            return Runner()

    result = extract_lease_details(DictsLLM(rows()), RESIDENTIAL)
    assert "sqft" in result.unexpected_fields
    assert result.details.tenant_name == "Priya Nair"


def test_all_null_result_is_flagged_as_extraction_failure():
    """The v1 bug this fixes: nine nulls looked identical whether the document
    was sparse or the extraction simply failed."""
    empty = ExtractionResult(
        fields=[ExtractedField(field=name) for name in FIELD_NAMES]
    )
    result = extract_lease_details(StrictLLM(empty), "hello world")
    assert result.no_fields_found is True
    assert len(result.missing) == len(FIELD_NAMES)


def test_schema_violating_row_raises_clear_error():
    bad = {"fields": [{"field": "not_a_real_field", "value": "x"}]}
    llm = StrictLLM(bad)
    with pytest.raises(ValueError, match="didn't match the expected schema"):
        extract_lease_details(llm, RESIDENTIAL)


# --- prompt ----------------------------------------------------------------


def test_prompt_lists_every_field_and_fences_the_document():
    llm = StrictLLM(rows())
    extract_lease_details(llm, RESIDENTIAL)
    prompt = llm.prompts[0]
    for name in FIELD_NAMES:
        assert name in prompt
    assert "<lease_document>" in prompt and "</lease_document>" in prompt


def test_prompt_tells_model_to_ignore_directions_inside_the_document():
    llm = StrictLLM(rows())
    extract_lease_details(llm, RESIDENTIAL)
    assert "untrusted" in llm.prompts[0].lower()


def test_structured_output_is_retried_on_transient_error():
    llm = StrictLLM(rows(), fail_first=2)
    result = extract_lease_details(llm, RESIDENTIAL)
    assert result.details.tenant_name == "Priya Nair"
    assert llm.calls == 3


# --- raw / salvage path ----------------------------------------------------


def test_raw_path_handles_bare_json():
    result = extract_lease_details(PlainLLM(flat_json()), RESIDENTIAL)
    assert result.details.monthly_rent == "Rs. 65,000"
    assert result.degraded is True


def test_raw_path_handles_fenced_json():
    fenced = f"```json\n{flat_json()}\n```"
    result = extract_lease_details(PlainLLM(fenced), RESIDENTIAL)
    assert result.details.tenant_name == "Priya Nair"


def test_raw_path_handles_prose_before_the_fence():
    """v1 raised here: the fence was only stripped when it started at char 0."""
    chatty = f"Sure! Here is the JSON you asked for:\n```json\n{flat_json()}\n```"
    result = extract_lease_details(PlainLLM(chatty), RESIDENTIAL)
    assert result.details.tenant_name == "Priya Nair"


def test_raw_path_handles_trailing_prose_after_the_fence():
    chatty = f"```json\n{flat_json()}\n```\nLet me know if you need the deposit split out."
    result = extract_lease_details(PlainLLM(chatty), RESIDENTIAL)
    assert result.details.landlord_name == "Arvind Shah"


def test_raw_path_understands_the_row_shape_too():
    payload = json.dumps({"fields": rows().model_dump()["fields"]})
    result = extract_lease_details(PlainLLM(payload), RESIDENTIAL)
    assert result.details.monthly_rent == "Rs. 65,000"
    assert result.evidence["monthly_rent"] == "monthly rent of Rs. 65,000"


def test_raw_path_surfaces_unexpected_keys():
    result = extract_lease_details(
        PlainLLM(flat_json(pet_policy="no cats")), RESIDENTIAL
    )
    assert "pet_policy" in result.unexpected_fields


def test_raw_path_rejects_prose_with_no_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        extract_lease_details(PlainLLM("I cannot help with that request."), "text")


def test_raw_path_rejects_a_json_array():
    with pytest.raises(ValueError, match="JSON array"):
        extract_lease_details(PlainLLM("[1, 2, 3]"), "text")


def test_raw_path_rejects_truncated_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        extract_lease_details(PlainLLM('{"tenant_name": "A", "landlord'), "text")


def test_wrong_type_for_a_field_fails_loudly():
    """v1 behaviour, deliberately kept: a bad type is a 422, not a silent cast."""
    payload = json.dumps({**json.loads(flat_json()), "tenant_name": 12345})
    with pytest.raises(ValueError):
        extract_lease_details(PlainLLM(payload), RESIDENTIAL)


# --- model shape -----------------------------------------------------------


def test_lease_details_dumps_all_nine_fields():
    data = LeaseDetails(tenant_name="Test").model_dump()
    assert len(data) == len(FIELD_NAMES) == 9
    assert data["landlord_name"] is None
