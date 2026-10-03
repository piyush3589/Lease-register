"""
API tests: status codes, size limits, token auth, and response shape.

The LLM is monkeypatched, so no API key or network access is needed.
"""

import pytest
from fastapi.testclient import TestClient

import api
from api import MAX_TEXT_CHARS, MAX_UPLOAD_BYTES
from documents import DocumentTooLarge, text_from_pdf_bytes
from extractor import (
    FIELD_NAMES,
    ExtractionResult,
    ExtractedField,
    LeaseDetails,
    LeaseExtraction,
)
from llm_utils import LLMCallError

client = TestClient(api.app)

GOOD_ROWS = ExtractionResult(
    fields=[
        ExtractedField(
            field="tenant_name",
            value="Priya Nair",
            source_quote="TENANT: Priya Nair",
        ),
        ExtractedField(field="monthly_rent", value="Rs. 65,000"),
    ]
)


@pytest.fixture(autouse=True)
def default_env(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_API_TOKEN", "")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch the LLM and the extraction call so tests never touch Groq."""
    monkeypatch.setattr(api, "get_llm", lambda: object())

    def install(result=None, error=None):
        def fake_extract(llm, text):
            if error:
                raise error
            return result

        monkeypatch.setattr(api, "extract_lease_details", fake_extract)

    return install


# --- health / readiness ----------------------------------------------------


def test_health_is_liveness_only():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_ok_even_without_api_key(monkeypatch):
    """Liveness must not depend on config; that is what /ready is for."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert client.get("/health").status_code == 200


def test_ready_reports_missing_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    body = client.get("/ready").json()
    assert body["status"] == "not_ready"
    assert "GROQ_API_KEY" in body["reason"]


def test_ready_reports_model_when_configured():
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert "model" in body


# --- extraction ------------------------------------------------------------


def test_extract_returns_fields_and_evidence(fake_llm):
    fake_llm(
        LeaseExtraction(
            details=LeaseDetails(tenant_name="Priya Nair", monthly_rent="Rs. 65,000"),
            evidence={"tenant_name": "TENANT: Priya Nair", "monthly_rent": None},
        )
    )
    response = client.post("/extract", json={"text": "TENANT: Priya Nair"})
    assert response.status_code == 200
    body = response.json()
    assert body["fields"]["tenant_name"] == "Priya Nair"
    assert body["fields"]["monthly_rent"] == "Rs. 65,000"
    assert body["fields"]["landlord_name"] is None
    assert body["evidence"]["tenant_name"] == "TENANT: Priya Nair"
    assert len(body["fields"]) == 9 == len(FIELD_NAMES)


def test_missing_fields_are_listed_explicitly(fake_llm):
    fake_llm(LeaseExtraction(details=LeaseDetails(tenant_name="A")))
    body = client.post("/extract", json={"text": "lease"}).json()
    assert body["missing_fields"] == [f for f in FIELD_NAMES if f != "tenant_name"]
    assert body["no_fields_found"] is False


def test_unverified_fields_are_surfaced_for_review(fake_llm):
    fake_llm(
        LeaseExtraction(
            details=LeaseDetails(monthly_rent="Rs. 1"),
            unverified_fields=["monthly_rent"],
        )
    )
    body = client.post("/extract", json={"text": "lease"}).json()
    assert body["unverified_fields"] == ["monthly_rent"]


def test_no_fields_found_flags_a_failed_extraction(fake_llm):
    fake_llm(LeaseExtraction(details=LeaseDetails()))
    body = client.post("/extract", json={"text": "hello"}).json()
    assert body["no_fields_found"] is True


def test_degraded_flag_is_passed_through(fake_llm):
    fake_llm(LeaseExtraction(details=LeaseDetails(tenant_name="A"), degraded=True))
    assert client.post("/extract", json={"text": "lease"}).json()["degraded"] is True


def test_whitespace_only_text_is_rejected(fake_llm):
    assert client.post("/extract", json={"text": "   "}).status_code == 400


def test_empty_text_is_rejected_by_schema(fake_llm):
    assert client.post("/extract", json={"text": ""}).status_code == 422


def test_missing_text_field_is_rejected(fake_llm):
    assert client.post("/extract", json={}).status_code == 422


def test_llm_failure_is_502(fake_llm):
    fake_llm(error=LLMCallError("rate limited"))
    response = client.post("/extract", json={"text": "lease text"})
    assert response.status_code == 502
    assert "rate limited" in response.json()["detail"]


def test_unusable_model_output_is_422(fake_llm):
    fake_llm(error=ValueError("Model output was not valid JSON"))
    assert client.post("/extract", json={"text": "lease text"}).status_code == 422


def test_missing_api_key_is_503(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    response = client.post("/extract", json={"text": "lease text"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


# --- size limits -----------------------------------------------------------


def test_oversize_text_is_413(fake_llm):
    response = client.post("/extract", json={"text": "x" * (MAX_TEXT_CHARS + 1)})
    assert response.status_code == 413
    assert f"{MAX_TEXT_CHARS:,}" in response.json()["detail"]


def test_text_at_the_limit_is_accepted(fake_llm):
    fake_llm(LeaseExtraction(details=LeaseDetails(tenant_name="A")))
    response = client.post("/extract", json={"text": "x" * MAX_TEXT_CHARS})
    assert response.status_code == 200


def test_oversize_upload_is_413(monkeypatch):
    """Guard against buffering an unbounded body. The cap is lowered rather than
    allocating a real 15 MB payload, to keep the suite fast."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 64)

    response = client.post(
        "/extract/file",
        files={"file": ("lease.pdf", b"%PDF-1.4" + b"\0" * 200, "application/pdf")},
    )
    assert response.status_code == 413


# --- auth ------------------------------------------------------------------


def test_missing_token_rejected_when_configured(monkeypatch, fake_llm):
    monkeypatch.setenv("EXTRACTOR_API_TOKEN", "secret-token")
    fake_llm(LeaseExtraction(details=LeaseDetails(tenant_name="A")))
    assert client.post("/extract", json={"text": "lease text"}).status_code == 401


def test_wrong_token_rejected_when_configured(monkeypatch, fake_llm):
    monkeypatch.setenv("EXTRACTOR_API_TOKEN", "secret-token")
    fake_llm(LeaseExtraction(details=LeaseDetails(tenant_name="A")))
    response = client.post(
        "/extract",
        json={"text": "lease text"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


def test_valid_token_allowed_when_configured(monkeypatch, fake_llm):
    monkeypatch.setenv("EXTRACTOR_API_TOKEN", "secret-token")
    fake_llm(LeaseExtraction(details=LeaseDetails(tenant_name="A")))
    response = client.post(
        "/extract",
        json={"text": "lease text"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 200


def test_file_endpoint_is_also_token_protected(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_API_TOKEN", "secret-token")
    response = client.post(
        "/extract/file", files={"file": ("lease.txt", b"x", "text/plain")}
    )
    assert response.status_code == 401


# --- file upload -----------------------------------------------------------


def test_non_pdf_upload_rejected():
    response = client.post(
        "/extract/file", files={"file": ("note.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_empty_pdf_upload_is_422():
    response = client.post("/extract/file", files={"file": ("lease.pdf", b"", "application/pdf")})
    assert response.status_code == 422


def test_unreadable_pdf_upload_is_422():
    response = client.post(
        "/extract/file",
        files={"file": ("lease.pdf", b"not a pdf at all", "application/pdf")},
    )
    assert response.status_code == 422


def test_oversize_extracted_text_is_413(monkeypatch, fake_llm):
    monkeypatch.setattr(
        api, "text_from_pdf_bytes", lambda raw: "y" * (MAX_TEXT_CHARS + 1)
    )
    fake_llm(LeaseExtraction(details=LeaseDetails(tenant_name="A")))
    response = client.post(
        "/extract/file", files={"file": ("lease.pdf", b"%PDF-1.4 fake", "application/pdf")}
    )
    assert response.status_code == 413


def test_document_too_large_maps_to_413(monkeypatch, fake_llm):
    def boom(raw):
        raise DocumentTooLarge("PDF has 900 pages; the limit is 60.")

    monkeypatch.setattr(api, "text_from_pdf_bytes", boom)
    fake_llm(LeaseExtraction(details=LeaseDetails(tenant_name="A")))
    response = client.post(
        "/extract/file", files={"file": ("lease.pdf", b"%PDF-1.4 fake", "application/pdf")}
    )
    assert response.status_code == 413


def test_valid_pdf_reaches_extraction(monkeypatch, fake_llm):
    monkeypatch.setattr(api, "text_from_pdf_bytes", lambda raw: "TENANT: Priya Nair")
    fake_llm(LeaseExtraction(details=LeaseDetails(tenant_name="Priya Nair")))
    response = client.post(
        "/extract/file",
        files={"file": ("lease.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["fields"]["tenant_name"] == "Priya Nair"


def test_real_sample_pdf_is_readable():
    """The generated sample PDFs must survive the real pypdf path, not just the
    monkeypatched one."""
    with open("sample_leases/lease_residential_01.pdf", "rb") as fh:
        text = text_from_pdf_bytes(fh.read())
    assert "Priya Nair" in text
