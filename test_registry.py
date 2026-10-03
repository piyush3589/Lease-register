"""
Tests for the SQLite lease register.

Each test points the module at its own file inside tmp_path, so the suite never
touches a real register and tests cannot see each other's rows.

The properties worth protecting are the ones a user would notice if broken:
re-running the same document must not create a second row, a failed attempt must
still be recorded, and a field the model could not verify must stay flagged
rather than quietly looking trustworthy.
"""

import csv
import io
import sqlite3

import pytest

import registry
from extractor import FIELD_NAMES, LeaseDetails, LeaseExtraction

RENT = "Rs. 65,000"
DEPOSIT = "Rs. 4,20,000"


@pytest.fixture(autouse=True)
def temp_registry(tmp_path, monkeypatch):
    """Point every operation at a throwaway file for this test."""
    db_path = tmp_path / "leases.db"
    monkeypatch.setenv("REGISTRY_DB", str(db_path))
    registry.init_db()
    return db_path


def db_file(db_path) -> str:
    return str(db_path)


def make_extraction(**overrides) -> LeaseExtraction:
    """A complete, fully verified extraction unless overridden."""
    details = LeaseDetails(
        tenant_name="Priya Sharma",
        landlord_name="Ramesh Iyer",
        property_address="14B, Nehru Colony, Pune 411014",
        lease_start_date="2026-01-01",
        lease_end_date="2026-12-31",
        lease_term="11 months",
        monthly_rent=RENT,
        security_deposit=DEPOSIT,
        renewal_terms="No automatic renewal",
    )
    payload = {
        "details": details,
        "evidence": {name: f"Supporting sentence for {name}." for name in FIELD_NAMES},
        "unverified_fields": [],
        "unexpected_fields": [],
        "degraded": False,
    }
    payload.update(overrides)
    return LeaseExtraction(**payload)


LEASE_TEXT = """
RENT AGREEMENT
This agreement is made on 1 January 2026 between Ramesh Iyer (landlord) and
Priya Sharma (tenant) for the premises at 14B, Nehru Colony, Pune 411014.
The tenancy runs from 1 January 2026 to 31 December 2026.
Monthly rent Rs. 65,000. Security deposit Rs. 4,20,000.
"""


# --- saving ----------------------------------------------------------------


def test_saving_returns_a_new_row():
    saved = registry.save_extraction(
        text=LEASE_TEXT,
        source_name="lease_a.txt",
        source_kind="text",
        extraction=make_extraction(),
    )
    assert saved["created"] is True
    assert saved["lease_id"] == 1
    assert saved["status"] == "ok"
    assert saved["needs_review"] is False


def test_all_fields_are_persisted():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    record = registry.get_lease(1)
    assert record["fields"]["tenant_name"] == "Priya Sharma"
    assert record["fields"]["monthly_rent"] == RENT
    assert record["fields"]["security_deposit"] == DEPOSIT
    assert record["evidence"]["monthly_rent"].startswith("Supporting sentence")


def test_missing_fields_stay_null_rather_than_empty_string():
    """Empty string and null mean different things downstream."""
    details = LeaseDetails(tenant_name="Asha Rao", monthly_rent=RENT)
    registry.save_extraction(
        text="partial lease", source_name="b.txt", source_kind="text",
        extraction=make_extraction(details=details, evidence={}),
    )
    record = registry.get_lease(1)
    assert record["fields"]["tenant_name"] == "Asha Rao"
    assert record["fields"]["landlord_name"] is None
    # The contract is a stable key set: callers should not have to guard.
    assert set(record["fields"]) == set(FIELD_NAMES)


def test_rent_is_parsed_into_a_summable_number():
    saved = registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert saved["monthly_rent_amount"] == 65000


def test_unparseable_rent_leaves_the_number_null_but_keeps_the_text():
    details = LeaseDetails(tenant_name="Asha Rao", monthly_rent="as mutually agreed")
    saved = registry.save_extraction(
        text="x", source_name="c.txt", source_kind="text",
        extraction=make_extraction(details=details, evidence={}),
    )
    assert saved["monthly_rent_amount"] is None
    assert registry.get_lease(1)["fields"]["monthly_rent"] == "as mutually agreed"


def test_pdf_source_kind_is_stored():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="lease.pdf", source_kind="pdf",
        extraction=make_extraction(),
    )
    assert registry.get_lease(1)["source_kind"] == "pdf"


# --- deduplication ---------------------------------------------------------


def test_same_document_does_not_create_a_second_row():
    first = registry.save_extraction(
        text=LEASE_TEXT, source_name="lease.txt", source_kind="text",
        extraction=make_extraction(),
    )
    second = registry.save_extraction(
        text=LEASE_TEXT, source_name="lease-copy.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert second["lease_id"] == first["lease_id"]
    assert second["created"] is False
    assert registry.count_leases() == 1


def test_re_extracting_updates_the_row_rather_than_skipping_it():
    """A corrected extraction must be able to replace a bad earlier one."""
    registry.save_extraction(
        text=LEASE_TEXT, source_name="lease.txt", source_kind="text",
        extraction=make_extraction(details=LeaseDetails(tenant_name="Wrong Name")),
    )
    updated = registry.save_extraction(
        text=LEASE_TEXT, source_name="lease.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert updated["created"] is False
    assert registry.get_lease(1)["fields"]["tenant_name"] == "Priya Sharma"


def test_whitespace_differences_are_not_a_new_lease():
    """Reformatting or re-saving a file must not duplicate a lease."""
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    again = registry.save_extraction(
        text=LEASE_TEXT + "\n\n   \n", source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert again["created"] is False
    assert registry.count_leases() == 1


def test_a_different_document_is_a_separate_lease():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    other = registry.save_extraction(
        text="A completely different lease about a flat in Mumbai.",
        source_name="b.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert other["lease_id"] == 2
    assert registry.count_leases() == 2


def test_content_hash_ignores_whitespace_and_case_of_spacing():
    assert registry.content_hash("a  b\nc") == registry.content_hash("a b \n  c")


def test_content_hash_differs_for_different_text():
    assert registry.content_hash("lease one") != registry.content_hash("lease two")


# --- review flags ----------------------------------------------------------


def test_unverified_field_is_flagged_and_survives_a_reload():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(unverified_fields=["monthly_rent"]),
    )
    record = registry.get_lease(1)
    assert record["needs_review"] is True
    assert "monthly_rent" in record["unverified_fields"]


def test_saving_nothing_is_flagged_for_review():
    """An empty extraction is a failure, not a lease with blank fields."""
    saved = registry.save_extraction(
        text="the purchase agreement, not a lease at all",
        source_name="bad.txt", source_kind="text",
        extraction=make_extraction(details=LeaseDetails(), evidence={}),
    )
    assert saved["needs_review"] is True
    assert bool(registry.get_lease(1)["no_fields_found"]) is True


# --- failures --------------------------------------------------------------


def test_a_failed_attempt_is_recorded():
    registry.record_failure(
        text="a document that broke the extractor",
        source_name="broken.pdf",
        source_kind="pdf",
        error="PDF has no extractable text layer",
    )
    record = registry.get_lease(1)
    assert record["status"] == "failed"
    assert "no extractable text" in record["error"]
    assert record["needs_review"] is True


def test_a_failure_is_counted_in_the_register():
    registry.record_failure(text="x", source_name="x", source_kind="pdf", error="boom")
    assert registry.count_leases() == 1
    assert registry.list_leases()[0]["status"] == "failed"


def test_retrying_a_failed_document_updates_the_same_row():
    registry.record_failure(
        text="half readable lease", source_name="f.txt", source_kind="text",
        error="model timeout",
    )
    retried = registry.save_extraction(
        text="half readable lease", source_name="f.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert retried["lease_id"] == 1
    assert retried["status"] == "ok"
    assert registry.count_leases() == 1


# --- reading ---------------------------------------------------------------


def test_missing_lease_reads_as_none():
    assert registry.get_lease(999) is None


def test_listing_is_newest_first_and_paginates():
    for i in range(5):
        registry.save_extraction(
            text=f"distinct lease document number {i}",
            source_name=f"lease_{i}.txt",
            source_kind="text",
            extraction=make_extraction(),
        )
    page = registry.list_leases(limit=2, offset=0)
    assert len(page) == 2
    assert page[0]["source_name"] == "lease_4.txt"

    second = registry.list_leases(limit=2, offset=2)
    assert second[0]["source_name"] == "lease_2.txt"


def test_listing_reports_field_counts_for_the_table():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(unverified_fields=["monthly_rent", "lease_term"]),
    )
    row = registry.list_leases()[0]
    assert row["fields_found"] == len(FIELD_NAMES)
    assert row["unverified_count"] == 2
    assert row["monthly_rent_amount"] == 65000


def test_search_matches_tenant():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    registry.save_extraction(
        text="a different agreement entirely", source_name="b.txt",
        source_kind="text",
        extraction=make_extraction(details=LeaseDetails(tenant_name="Rahul Verma")),
    )
    assert len(registry.list_leases(search="Priya")) == 1
    assert registry.list_leases(search="Priya")[0]["tenant_name"] == "Priya Sharma"


def test_search_matches_filename():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="banalore_flat.pdf", source_kind="pdf",
        extraction=make_extraction(),
    )
    assert len(registry.list_leases(search="banalore")) == 1


def test_search_matches_any_field_value():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert len(registry.list_leases(search="Nehru Colony")) == 1


def test_search_that_matches_nothing_returns_empty():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert registry.list_leases(search="no such tenant") == []


def test_data_survives_reopening_the_database(temp_registry, monkeypatch):
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    # Re-point at the same file and reconnect from scratch.
    monkeypatch.setenv("REGISTRY_DB", db_file(temp_registry))
    registry.init_db()
    assert registry.count_leases() == 1
    assert registry.get_lease(1)["fields"]["tenant_name"] == "Priya Sharma"


# --- deleting --------------------------------------------------------------


def test_deleting_removes_the_lease_and_its_fields():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert registry.delete_lease(1) is True
    assert registry.get_lease(1) is None
    assert registry.count_leases() == 0


def test_deleting_leaves_no_orphan_field_rows(temp_registry):
    """A deleted lease must not keep its field values lying around in the DB."""
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    registry.delete_lease(1)
    with sqlite3.connect(db_file(temp_registry)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM lease_fields").fetchone()[0] == 0


def test_deleting_twice_is_not_an_error():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    assert registry.delete_lease(1) is True
    assert registry.delete_lease(1) is False


def test_deleting_one_lease_leaves_others_alone():
    registry.save_extraction(
        text="document one", source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    registry.save_extraction(
        text="document two", source_name="b.txt", source_kind="text",
        extraction=make_extraction(),
    )
    registry.delete_lease(1)
    assert registry.get_lease(2) is not None
    assert registry.count_leases() == 1


# --- CSV -------------------------------------------------------------------


def test_csv_has_a_header_and_one_row_per_lease():
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    rows = list(csv.DictReader(io.StringIO(registry.export_csv())))
    assert len(rows) == 1
    assert rows[0]["tenant_name"] == "Priya Sharma"
    assert rows[0]["monthly_rent_amount"] == "65000"


def test_csv_includes_every_field_even_when_absent():
    registry.save_extraction(
        text="partial", source_name="p.txt", source_kind="text",
        extraction=make_extraction(details=LeaseDetails(tenant_name="Asha Rao")),
    )
    rows = list(csv.DictReader(io.StringIO(registry.export_csv())))
    assert rows[0]["landlord_name"] == ""


def test_empty_register_still_exports_a_usable_header():
    """An empty CSV must open cleanly in Excel, header row and all."""
    text = registry.export_csv()
    header = next(csv.reader(io.StringIO(text)))
    assert "lease_id" in header
    assert "tenant_name" in header
    assert list(csv.DictReader(io.StringIO(text))) == []


def test_csv_escapes_values_containing_commas():
    """Rent strings are full of commas; unquoted they would split the columns."""
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    rows = list(csv.DictReader(io.StringIO(registry.export_csv())))
    assert rows[0]["monthly_rent"] == RENT
    assert rows[0]["security_deposit"] == DEPOSIT


def test_csv_includes_failed_attempts():
    registry.record_failure(text="x", source_name="x.pdf", source_kind="pdf", error="boom")
    rows = list(csv.DictReader(io.StringIO(registry.export_csv())))
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "boom"


# --- schema ----------------------------------------------------------------


def test_init_db_is_idempotent():
    registry.init_db()
    registry.init_db()
    assert registry.count_leases() == 0


def test_foreign_keys_cascade_are_declared(temp_registry):
    """Belt and braces alongside the manual delete."""
    registry.save_extraction(
        text=LEASE_TEXT, source_name="a.txt", source_kind="text",
        extraction=make_extraction(),
    )
    with sqlite3.connect(db_file(temp_registry)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM leases WHERE id = 1")
        assert conn.execute("SELECT COUNT(*) FROM lease_fields").fetchone()[0] == 0