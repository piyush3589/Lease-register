"""
End-to-end: real sample lease text, through the real API, with a scripted model.

Unit tests check each piece. This checks the wiring -- prompt building, the
provenance check against actual document text, the response envelope, and the
UI-facing flags -- which is where a refactor like v1->v2 actually breaks.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
from extractor import FIELD_NAMES, ExtractionResult, ExtractedField, LeaseExtraction

client = TestClient(api.app)

SAMPLES = Path("sample_leases")


def read_sample(name: str) -> str:
    return (SAMPLES / f"{name}.txt").read_text(encoding="utf-8")


class ScriptedLLM:
    """Returns a fixed ExtractionResult, and records the prompt it was given."""

    def __init__(self, result):
        self.result = result
        self.prompts = []

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class Runner:
            def invoke(self, messages):
                outer.prompts = [m.content for m in messages]
                return outer.result

        return Runner()


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_API_TOKEN", "")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def install(monkeypatch, llm):
    monkeypatch.setattr(api, "get_llm", lambda: llm)
    monkeypatch.setattr(api, "extract_lease_details", lambda _llm, text: llm.result)


def test_complete_lease_end_to_end(monkeypatch):
    text = read_sample("lease_residential_01")
    # Quotes copied from sample_leases/lease_residential_01.txt, which states
    # parties in prose ("Ms. Priya Nair (Tenant)") rather than as labelled lines.
    result = LeaseExtraction(
        details=api.LeaseDetails(
            tenant_name="Priya Nair",
            landlord_name="Arvind Shah",
            property_address=(
                "Flat No. 1204, Oberoi Splendor, JVLR, Andheri East, "
                "Mumbai, Maharashtra 400060"
            ),
            lease_start_date="1st April 2025",
            lease_end_date="28th February 2026",
            lease_term="11 months",
            monthly_rent="Rs. 65,000",
            security_deposit="Rs. 3,25,000",
            renewal_terms="renewed for a further period upon mutual written consent",
        ),
        evidence={
            "tenant_name": "Ms. Priya Nair (Tenant)",
            "landlord_name": "Mr. Arvind Shah (Landlord)",
            "property_address": (
                "Flat No. 1204, Oberoi Splendor, JVLR, Andheri East, "
                "Mumbai, Maharashtra 400060"
            ),
            "lease_start_date": "commencing from 1st April 2025",
            "lease_end_date": "ending on 28th February 2026",
            "lease_term": "valid for a period of 11 months",
            "monthly_rent": "monthly rent of Rs. 65,000",
            "security_deposit": "refundable security deposit of Rs. 3,25,000",
            "renewal_terms": (
                "renewed for a further period upon mutual written consent"
            ),
        },
    )
    install(monkeypatch, ScriptedLLM(result))

    body = client.post("/extract", json={"text": text}).json()

    assert body["fields"]["tenant_name"] == "Priya Nair"
    assert body["fields"]["monthly_rent"] == "Rs. 65,000"
    assert body["missing_fields"] == []
    assert body["unverified_fields"] == [], "quotes from the real document must verify"
    assert body["no_fields_found"] is False
    assert body["degraded"] is False
    assert len(body["fields"]) == 9


def test_incomplete_lease_returns_nulls_not_inventions(monkeypatch):
    """The headline demo: an informal note has no landlord or dates, and the
    response must say so rather than filling them in."""
    text = read_sample("lease_incomplete_01")
    result = LeaseExtraction(
        details=api.LeaseDetails(
            tenant_name="Rohan Mehta",
            property_address="2BHK, Hiranandani Gardens, Powai, Mumbai",
            monthly_rent="Rs. 48,000",
        ),
        evidence={
            "tenant_name": "Tenant: Rohan Mehta",
            "monthly_rent": "Rent of Rs. 48,000 per month agreed",
        },
    )
    install(monkeypatch, ScriptedLLM(result))

    body = client.post("/extract", json={"text": text}).json()

    assert body["fields"]["landlord_name"] is None
    assert body["fields"]["lease_start_date"] is None
    assert body["fields"]["renewal_terms"] is None
    assert "landlord_name" in body["missing_fields"]
    assert "renewal_terms" in body["missing_fields"]
    assert body["no_fields_found"] is False
    assert body["unverified_fields"] == []


def test_fabricated_value_is_flagged_against_the_real_document(monkeypatch):
    """A value with no support in the actual sample text must not pass as
    verified -- this is the check that makes the output trustworthy.

    Unlike the tests above, this one lets the real extractor run (only get_llm is
    stubbed). Stubbing extract_lease_details would skip verification entirely and
    assert nothing about it.
    """
    text = read_sample("lease_residential_01")
    llm = ScriptedLLM(
        ExtractionResult(
            fields=[
                ExtractedField(
                    field="monthly_rent",
                    value="Rs. 1",
                    source_quote="monthly rent of Rs. 65,000",
                )
            ]
        )
    )
    monkeypatch.setattr(api, "get_llm", lambda: llm)

    body = client.post("/extract", json={"text": text}).json()

    assert body["fields"]["monthly_rent"] == "Rs. 1", "the value is still returned"
    assert body["unverified_fields"] == ["monthly_rent"], "but not as verified"


def test_quote_not_in_the_document_is_flagged(monkeypatch):
    """A quote the source text does not contain is the strongest fabrication
    signal there is."""
    text = read_sample("lease_residential_01")
    llm = ScriptedLLM(
        ExtractionResult(
            fields=[
                ExtractedField(
                    field="tenant_name",
                    value="Priya Nair",
                    source_quote="TENANT: Priya Nair",
                )
            ]
        )
    )
    monkeypatch.setattr(api, "get_llm", lambda: llm)

    body = client.post("/extract", json={"text": text}).json()
    assert body["unverified_fields"] == ["tenant_name"]


def test_real_extractor_flags_a_bad_quote_on_a_real_sample(monkeypatch):
    """The prose wording of the sample ("Ms. Priya Nair (Tenant)") must verify,
    which pins the normalisation behaviour against real document text."""
    text = read_sample("lease_residential_01")
    llm = ScriptedLLM(
        ExtractionResult(
            fields=[
                ExtractedField(
                    field="tenant_name",
                    value="Priya Nair",
                    source_quote="Ms. Priya Nair (Tenant)",
                ),
                ExtractedField(
                    field="security_deposit",
                    value="Rs. 3,25,000",
                    source_quote="refundable security deposit of Rs. 3,25,000",
                ),
            ]
        )
    )
    monkeypatch.setattr(api, "get_llm", lambda: llm)

    body = client.post("/extract", json={"text": text}).json()
    assert body["unverified_fields"] == []
    assert body["degraded"] is False


def test_prompt_delivers_the_real_document_fenced(monkeypatch):
    text = read_sample("lease_commercial_01")
    llm = ScriptedLLM(LeaseExtraction(details=api.LeaseDetails(tenant_name="X")))
    monkeypatch.setattr(api, "get_llm", lambda: llm)
    captured = {}

    def fake_extract(_llm, sent_text):
        captured["text"] = sent_text
        return llm.result

    monkeypatch.setattr(api, "extract_lease_details", fake_extract)
    client.post("/extract", json={"text": text})

    assert captured["text"] == text.strip()


def test_full_extraction_prompt_contains_document_and_all_fields(monkeypatch):
    """Exercise the real extractor (not a stub) with a scripted model, so the
    prompt template and the provenance check are covered together."""
    from extractor import extract_lease_details

    text = read_sample("lease_residential_01")
    llm = ScriptedLLM(
        ExtractionResult(
            fields=[
                ExtractedField(
                    field="tenant_name",
                    value="Priya Nair",
                    source_quote="Ms. Priya Nair (Tenant)",
                ),
                ExtractedField(
                    field="monthly_rent",
                    value="Rs. 65,000",
                    source_quote="monthly rent of Rs. 65,000",
                ),
            ]
        )
    )

    extraction = extract_lease_details(llm, text)

    assert extraction.details.tenant_name == "Priya Nair"
    assert extraction.unverified_fields == []
    assert extraction.degraded is False
    assert set(extraction.evidence) == set(FIELD_NAMES)
    assert len(llm.prompts) == 1
    assert "<lease_document>" in llm.prompts[0]
    for name in FIELD_NAMES:
        assert name in llm.prompts[0]


def test_sample_pdfs_round_trip_through_the_file_endpoint(monkeypatch):
    """The PDF path, end to end, on generated fixtures."""
    for name in (
        "lease_residential_01",
        "lease_commercial_01",
        "lease_incomplete_01",
    ):
        llm = ScriptedLLM(
            LeaseExtraction(details=api.LeaseDetails(tenant_name="Someone"))
        )
        install(monkeypatch, llm)
        raw = (SAMPLES / f"{name}.pdf").read_bytes()
        response = client.post(
            "/extract/file", files={"file": (f"{name}.pdf", raw, "application/pdf")}
        )
        assert response.status_code == 200, f"{name}: {response.text}"
        assert response.json()["fields"]["tenant_name"] == "Someone"
