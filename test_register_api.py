"""
Tests for the register endpoints: /leases*, and the promise that /extract stays
stateless.

The LLM is monkeypatched and the database is a throwaway file per test, so this
covers the whole path from request to stored row without a network call.
"""

import csv
import io

import pytest
from fastapi.testclient import TestClient

import api
import registry
from extractor import FIELD_NAMES, LeaseDetails, LeaseExtraction

client = TestClient(api.app)

LEASE_TEXT = """
RENT AGREEMENT between Ramesh Iyer (landlord) and Priya Sharma (tenant) for
14B, Nehru Colony, Pune 411014, from 1 January 2026 to 31 December 2026.
Monthly rent Rs. 65,000. Security deposit Rs. 4,20,000.
"""


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("EXTRACTOR_API_TOKEN", "")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("REGISTRY_DB", str(tmp_path / "leases.db"))
    registry.init_db()


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(api, "get_llm", lambda: object())

    def install(result=None, error=None):
        def fake_extract(llm, text):
            if error:
                raise error
            return result

        monkeypatch.setattr(api, "extract_lease_details", fake_extract)

    return install


def complete_extraction(tenant: str = "Priya Sharma") -> LeaseExtraction:
    """An extraction whose evidence actually contains the values it claims."""
    details = LeaseDetails(
        tenant_name=tenant,
        landlord_name="Ramesh Iyer",
        property_address="14B, Nehru Colony, Pune 411014",
        lease_start_date="2026-01-01",
        lease_end_date="2026-12-31",
        lease_term="11 months",
        monthly_rent="Rs. 65,000",
        security_deposit="Rs. 4,20,000",
        renewal_terms="No automatic renewal",
    )
    return LeaseExtraction(
        details=details,
        evidence={
            "tenant_name": f"the tenant is {tenant}",
            "landlord_name": "the landlord is Ramesh Iyer",
            "property_address": "the premises at 14B, Nehru Colony, Pune 411014",
            "lease_start_date": "commencing 1 January 2026",
            "lease_end_date": "ending 31 December 2026",
            "lease_term": "for a term of 11 months",
            "monthly_rent": "Monthly rent Rs. 65,000",
            "security_deposit": "Security deposit Rs. 4,20,000",
            "renewal_terms": "There is no automatic renewal.",
        },
    )


def post_lease(client, text=LEASE_TEXT, name="lease.txt", headers=None):
    payload = {"text": text, "source_name": name}
    return client.post("/leases", json=payload, headers=headers)


# --- /extract stays stateless ----------------------------------------------


def test_extract_does_not_write_to_the_register(fake_llm):
    """The reason /extract exists: retrying one document must not create rows."""
    fake_llm(complete_extraction())
    assert client.post("/extract", json={"text": LEASE_TEXT}).status_code == 200
    assert client.post("/extract", json={"text": LEASE_TEXT}).status_code == 200
    assert registry.count_leases() == 0


def test_extract_response_is_unchanged_by_the_register_work(fake_llm):
    """Backwards compatibility for anything already calling /extract."""
    fake_llm(complete_extraction())
    body = client.post("/extract", json={"text": LEASE_TEXT}).json()
    assert set(body) == {
        "fields", "evidence", "missing_fields", "unverified_fields",
        "unexpected_fields", "no_fields_found", "degraded",
    }
    assert body["fields"]["tenant_name"] == "Priya Sharma"


# --- POST /leases ----------------------------------------------------------


def test_saving_a_lease_returns_its_id_and_parsed_rent(fake_llm):
    fake_llm(complete_extraction())
    response = post_lease(client)
    assert response.status_code == 201

    body = response.json()
    assert body["lease_id"] == 1
    assert body["created"] is True
    assert body["needs_review"] is False
    assert body["monthly_rent_amount"] == 65000


def test_saving_carries_the_full_extraction_in_the_response(fake_llm):
    """The client should not need a second request to show the result."""
    fake_llm(complete_extraction())
    body = post_lease(client).json()
    assert body["extraction"]["fields"]["monthly_rent"] == "Rs. 65,000"
    assert "Rs. 65,000" in body["extraction"]["evidence"]["monthly_rent"]


def test_reposting_the_same_lease_does_not_create_a_duplicate(fake_llm):
    fake_llm(complete_extraction())
    first = post_lease(client).json()
    second = post_lease(client).json()
    assert second["lease_id"] == first["lease_id"]
    assert second["created"] is False
    assert registry.count_leases() == 1


def test_a_different_lease_becomes_a_second_row(fake_llm):
    fake_llm(complete_extraction())
    post_lease(client)
    other = post_lease(client, text="A totally different agreement about a flat.")
    assert other.json()["lease_id"] == 2
    assert registry.count_leases() == 2


def test_source_kind_defaults_to_text_and_can_be_declared(fake_llm):
    fake_llm(complete_extraction())
    post_lease(client)
    client.post(
        "/leases",
        json={"text": "another lease entirely", "source_name": "x.pdf",
              "source_kind": "pdf"},
    )
    assert {r["source_kind"] for r in registry.list_leases()} == {"pdf", "text"}


def test_an_unknown_source_kind_is_rejected(fake_llm):
    fake_llm(complete_extraction())
    response = client.post(
        "/leases", json={"text": "x", "source_name": "y", "source_kind": "docx"}
    )
    assert response.status_code == 422


def test_blank_text_is_rejected_before_any_extraction(fake_llm):
    fake_llm(complete_extraction())
    assert post_lease(client, text="   ").status_code == 400
    assert registry.count_leases() == 0


def test_oversize_text_is_rejected_with_413(fake_llm):
    fake_llm(complete_extraction())
    response = post_lease(client, text="a" * (api.MAX_TEXT_CHARS + 1))
    assert response.status_code == 413
    assert registry.count_leases() == 0


def test_an_unparseable_rent_saves_with_a_null_amount(fake_llm):
    """Never refuse to save because of an odd rent string; keep the verbatim."""
    extraction = complete_extraction()
    extraction.details.monthly_rent = "to be mutually agreed"
    fake_llm(extraction)

    body = post_lease(client).json()
    assert body["monthly_rent_amount"] is None
    assert registry.get_lease(1)["fields"]["monthly_rent"] == "to be mutually agreed"


def test_an_unverified_value_is_flagged_for_review(fake_llm):
    extraction = complete_extraction()
    extraction.unverified_fields = ["security_deposit"]
    fake_llm(extraction)

    body = post_lease(client).json()
    assert body["needs_review"] is True
    assert registry.get_lease(1)["needs_review"] is True


# --- failures are remembered ----------------------------------------------


def test_a_model_failure_returns_502_and_saves_nothing(fake_llm, monkeypatch):
    from llm_utils import LLMCallError
    fake_llm(error=LLMCallError("provider timed out"))
    response = post_lease(client)
    assert response.status_code == 502
    # Nothing to hash-and-store reliably here: the caller never got a result.
    assert registry.count_leases() == 0


# --- auth ------------------------------------------------------------------


def test_register_endpoints_require_the_token_when_one_is_set(fake_llm, monkeypatch):
    fake_llm(complete_extraction())
    monkeypatch.setenv("EXTRACTOR_API_TOKEN", "s3cret")
    headers = {"Authorization": "Bearer s3cret"}

    assert post_lease(client).status_code == 401
    assert post_lease(client, name="a.txt", headers=headers).status_code == 201
    assert client.get("/leases").status_code == 401
    assert client.get("/leases", headers=headers).status_code == 200
    assert client.get("/leases.csv", headers=headers).status_code == 200
    assert client.get("/leases/1", headers=headers).status_code == 200
    assert client.delete("/leases/1", headers=headers).status_code == 200


def test_a_wrong_token_is_refused(fake_llm, monkeypatch):
    fake_llm(complete_extraction())
    monkeypatch.setenv("EXTRACTOR_API_TOKEN", "s3cret")
    response = client.get(
        "/leases", headers={"Authorization": "Bearer wrong"}
    )
    assert response.status_code == 401


# --- GET /leases -----------------------------------------------------------


def test_listing_is_empty_before_anything_is_saved():
    body = client.get("/leases").json()
    assert body == {"total": 0, "leases": []}


def test_listing_reports_total_and_rows(fake_llm):
    fake_llm(complete_extraction())
    post_lease(client)
    post_lease(client, text="a second, unrelated lease document", name="b.txt")

    body = client.get("/leases").json()
    assert body["total"] == 2
    assert len(body["leases"]) == 2
    assert body["leases"][0]["source_name"] == "b.txt"


def test_listing_can_be_searched(fake_llm, monkeypatch):
    fake_llm(complete_extraction())
    post_lease(client)

    # A second lease with a genuinely different tenant, so the search has
    # something to exclude.
    monkeypatch.setattr(
        api,
        "extract_lease_details",
        lambda llm, text: complete_extraction(tenant="Rahul Verma"),
    )
    post_lease(client, text="another agreement for a different tenant", name="b.txt")

    assert client.get("/leases", params={"search": "Priya"}).json()["total"] == 1
    assert client.get("/leases", params={"search": "Rahul"}).json()["total"] == 1
    assert client.get("/leases", params={"search": "nobody"}).json()["total"] == 0


def test_listing_paginates(fake_llm):
    fake_llm(complete_extraction())
    for i in range(3):
        post_lease(client, text=f"lease document number {i}", name=f"lease_{i}.txt")

    body = client.get("/leases", params={"limit": 2, "offset": 0}).json()
    assert len(body["leases"]) == 2
    assert body["total"] == 3


def test_pagination_bounds_are_validated():
    assert client.get("/leases", params={"limit": 0}).status_code == 422
    assert client.get("/leases", params={"limit": 5000}).status_code == 422
    assert client.get("/leases", params={"offset": -1}).status_code == 422


# --- GET /leases/{id} ------------------------------------------------------


def test_detail_returns_fields_and_evidence(fake_llm):
    fake_llm(complete_extraction())
    post_lease(client)

    body = client.get("/leases/1").json()
    assert body["id"] == 1
    assert body["fields"]["tenant_name"] == "Priya Sharma"
    assert set(body["fields"]) == set(FIELD_NAMES)
    assert "Priya" in body["evidence"]["tenant_name"]


def test_detail_for_an_unknown_id_is_404():
    response = client.get("/leases/4242")
    assert response.status_code == 404
    assert "4242" in response.json()["detail"]


def test_a_non_numeric_id_is_422():
    assert client.get("/leases/not-a-number").status_code == 422


# --- CSV -------------------------------------------------------------------


def test_csv_downloads_as_an_attachment(fake_llm):
    fake_llm(complete_extraction())
    post_lease(client)

    response = client.get("/leases.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "lease_register.csv" in response.headers["content-disposition"]


def test_csv_contains_the_saved_lease(fake_llm):
    fake_llm(complete_extraction())
    post_lease(client)

    rows = list(csv.DictReader(io.StringIO(client.get("/leases.csv").text)))
    assert len(rows) == 1
    assert rows[0]["tenant_name"] == "Priya Sharma"
    assert rows[0]["monthly_rent_amount"] == "65000"
    assert rows[0]["status"] == "ok"


def test_csv_route_is_not_swallowed_by_the_id_route(fake_llm):
    """A literal path declared after /leases/{id} would 422 on every download."""
    fake_llm(complete_extraction())
    assert client.get("/leases.csv").status_code == 200


# --- DELETE ----------------------------------------------------------------


def test_deleting_removes_the_lease(fake_llm):
    fake_llm(complete_extraction())
    post_lease(client)

    assert client.delete("/leases/1").status_code == 200
    assert registry.count_leases() == 0
    assert client.get("/leases/1").status_code == 404


def test_deleting_twice_is_404_the_second_time(fake_llm):
    fake_llm(complete_extraction())
    post_lease(client)
    client.delete("/leases/1")
    assert client.delete("/leases/1").status_code == 404