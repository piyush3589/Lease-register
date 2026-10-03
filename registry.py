"""
registry.py
The lease register: a local SQLite file that remembers every extraction.

This is what turns a one-shot demo into a tool. Without it, extracting the same
lease twice gives you nothing the second time and closing the tab loses
everything. With it, the register accumulates and becomes the thing operations
actually maintain.

Three decisions worth knowing about:

Deduplication is by document content, not filename. The same lease arriving as
"lease_final.pdf" and "lease_final_v2.pdf" is one record, not two. Re-running an
extraction updates that record in place rather than appending a near-duplicate,
so the register stays trustworthy as a record of what is actually let.

Failures are recorded too. A lease that could not be extracted gets a row with
status='failed' and the error, so a batch run leaves a visible trail instead of
silently dropping files.

The extracted string values are never rewritten. `monthly_rent_amount` is a
separate derived column, null whenever the text was not confidently a number.
"""

import csv
import hashlib
import io
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from amounts import parse_amount
from extractor import FIELD_NAMES

DEFAULT_DB_PATH = "lease_register.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS leases (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash        TEXT    NOT NULL UNIQUE,
    source_name         TEXT    NOT NULL,
    source_kind         TEXT    NOT NULL CHECK (source_kind IN ('pdf', 'text')),
    status              TEXT    NOT NULL CHECK (status IN ('ok', 'failed')),
    error               TEXT,
    degraded            INTEGER NOT NULL DEFAULT 0,
    no_fields_found     INTEGER NOT NULL DEFAULT 0,
    fields_found        INTEGER NOT NULL DEFAULT 0,
    unverified_count    INTEGER NOT NULL DEFAULT 0,
    monthly_rent_amount REAL,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS lease_fields (
    lease_id     INTEGER NOT NULL REFERENCES leases(id) ON DELETE CASCADE,
    field_name   TEXT    NOT NULL,
    value        TEXT,
    source_quote TEXT,
    verified     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (lease_id, field_name)
);

CREATE INDEX IF NOT EXISTS idx_leases_updated ON leases(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fields_name ON lease_fields(field_name);
"""


def db_path() -> str:
    """Registry location. Override with REGISTRY_DB."""
    return os.getenv("REGISTRY_DB", DEFAULT_DB_PATH)


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    """A fresh connection with foreign keys enforced.

    Deliberately not a shared connection: FastAPI runs sync endpoints on a
    threadpool, and sqlite3 connections are not safe to use across threads.
    Opening per call is cheap for a file this size.
    """
    conn = sqlite3.connect(path or db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Optional[str] = None) -> None:
    with session(path) as conn:
        conn.executescript(SCHEMA)


def content_hash(text: str) -> str:
    """Stable id for a document's text.

    Whitespace is folded first because PDF text extraction is not byte-stable
    across runs in a way that matters to a reader, and a document that renders
    with different line breaks is still the same lease. Case is preserved --
    two documents differing only in case are not assumed to be the same one.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _needs_review(row_fields: int, unverified: int, status: str, none_found: bool) -> bool:
    return status != "ok" or none_found or unverified > 0 or row_fields < len(FIELD_NAMES)


def save_extraction(
    *,
    text: str,
    source_name: str,
    source_kind: str,
    extraction=None,
    error: Optional[str] = None,
    path: Optional[str] = None,
) -> dict:
    """Record one extraction. Returns {lease_id, created, status}.

    When the same document text has been seen before, the existing row is
    updated instead of a second row being added. `created` says which happened,
    so the caller can tell the user "already in the register, refreshed" rather
    than implying a new lease was discovered.
    """
    if source_kind not in ("pdf", "text"):
        raise ValueError("source_kind must be 'pdf' or 'text'")

    init_db(path)

    digest = content_hash(text)
    status = "failed" if extraction is None else "ok"

    fields = extraction.details.model_dump() if extraction is not None else {}
    evidence = (extraction.evidence if extraction is not None else {}) or {}
    unverified = set(extraction.unverified_fields if extraction is not None else [])
    none_found = extraction.no_fields_found if extraction is not None else True
    degraded = extraction.degraded if extraction is not None else False
    fields_found = sum(1 for v in fields.values() if v is not None)

    rent_amount = parse_amount(fields.get("monthly_rent"))
    needs_review = _needs_review(fields_found, len(unverified), status, none_found)
    now = _now()

    with session(path) as conn:
        existing = conn.execute(
            "SELECT id, created_at FROM leases WHERE content_hash = ?", (digest,)
        ).fetchone()

        if existing:
            lease_id = existing["id"]
            created = False
            conn.execute(
                """UPDATE leases
                      SET source_name = ?, source_kind = ?, status = ?, error = ?,
                          degraded = ?, no_fields_found = ?, fields_found = ?,
                          unverified_count = ?, monthly_rent_amount = ?,
                          updated_at = ?
                    WHERE id = ?""",
                (
                    source_name, source_kind, status, error,
                    int(degraded), int(none_found), fields_found,
                    len(unverified), rent_amount, now, lease_id,
                ),
            )
            conn.execute("DELETE FROM lease_fields WHERE lease_id = ?", (lease_id,))
        else:
            cursor = conn.execute(
                """INSERT INTO leases
                       (content_hash, source_name, source_kind, status, error,
                        degraded, no_fields_found, fields_found, unverified_count,
                        monthly_rent_amount, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    digest, source_name, source_kind, status, error,
                    int(degraded), int(none_found), fields_found,
                    len(unverified), rent_amount, now, now,
                ),
            )
            lease_id = cursor.lastrowid
            created = True

        if extraction is not None:
            conn.executemany(
                """INSERT INTO lease_fields (lease_id, field_name, value, source_quote, verified)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        lease_id,
                        name,
                        fields.get(name),
                        evidence.get(name),
                        0 if name in unverified else 1,
                    )
                    for name in FIELD_NAMES
                ],
            )

    return {
        "lease_id": lease_id,
        "created": created,
        "status": status,
        "needs_review": needs_review,
        "monthly_rent_amount": rent_amount,
    }


def record_failure(
    *, text: str, source_name: str, source_kind: str, error: str, path: Optional[str] = None
) -> dict:
    """Record a lease that could not be extracted, so the attempt is visible."""
    return save_extraction(
        text=text,
        source_name=source_name,
        source_kind=source_kind,
        extraction=None,
        error=error,
        path=path,
    )


def list_leases(
    path: Optional[str] = None, limit: int = 100, offset: int = 0, search: str = ""
) -> list[dict]:
    """Newest first. `search` matches source name or any extracted value."""
    # Timestamps have one-second resolution, so a burst of saves would tie.
    # The id tiebreak keeps paging stable instead of shuffling rows arbitrarily.
    ORDER_BY = "ORDER BY l.updated_at DESC, l.id DESC"

    init_db(path)
    clauses = []
    params: list = []

    if search:
        clauses.append(
            "(l.source_name LIKE ? OR EXISTS ("
            "  SELECT 1 FROM lease_fields f"
            "  WHERE f.lease_id = l.id AND f.value LIKE ?))"
        )
        params.extend([f"%{search}%", f"%{search}%"])

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([limit, offset])

    with session(path) as conn:
        rows = conn.execute(
            f"""SELECT l.*,
                       (SELECT f.value FROM lease_fields f
                         WHERE f.lease_id = l.id AND f.field_name = 'tenant_name')  AS tenant_name,
                       (SELECT f.value FROM lease_fields f
                         WHERE f.lease_id = l.id AND f.field_name = 'monthly_rent') AS monthly_rent
                  FROM leases l
                  {where}
                 {ORDER_BY}, l.id DESC
                 LIMIT ? OFFSET ?""",
            params,
        ).fetchall()

    return [dict(row) for row in rows]


def count_leases(path: Optional[str] = None, search: str = "") -> int:
    """How many leases match. Pass the same `search` the caller listed with, or
    the count will disagree with the rows it just returned."""
    init_db(path)
    if search:
        with session(path) as conn:
            return conn.execute(
                "SELECT COUNT(*) AS n FROM leases l WHERE "
                "(l.source_name LIKE ? OR EXISTS ("
                "  SELECT 1 FROM lease_fields f"
                "  WHERE f.lease_id = l.id AND f.value LIKE ?))",
                (f"%{search}%", f"%{search}%"),
            ).fetchone()["n"]

    with session(path) as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM leases").fetchone()["n"]


def get_lease(lease_id: int, path: Optional[str] = None) -> Optional[dict]:
    """One lease with its fields, evidence, and which values are unverified."""
    init_db(path)
    with session(path) as conn:
        row = conn.execute("SELECT * FROM leases WHERE id = ?", (lease_id,)).fetchone()
        if row is None:
            return None
        field_rows = conn.execute(
            "SELECT * FROM lease_fields WHERE lease_id = ?", (lease_id,)
        ).fetchall()

    fields = {r["field_name"]: r["value"] for r in field_rows}
    evidence = {r["field_name"]: r["source_quote"] for r in field_rows}
    unverified = [r["field_name"] for r in field_rows if not r["verified"]]

    record = dict(row)
    record["fields"] = {name: fields.get(name) for name in FIELD_NAMES}
    record["evidence"] = {name: evidence.get(name) for name in FIELD_NAMES}
    record["unverified_fields"] = unverified
    record["missing_fields"] = [n for n in FIELD_NAMES if fields.get(n) is None]
    record["needs_review"] = _needs_review(
        record["fields_found"], record["unverified_count"], record["status"],
        bool(record["no_fields_found"]),
    )
    return record


def delete_lease(lease_id: int, path: Optional[str] = None) -> bool:
    init_db(path)
    with session(path) as conn:
        cursor = conn.execute("DELETE FROM leases WHERE id = ?", (lease_id,))
        return cursor.rowcount > 0


CSV_COLUMNS = [
    "lease_id", "source_name", "source_kind", "status", "needs_review",
    "created_at", "updated_at", "fields_found", "unverified_count",
    "no_fields_found", "error", "monthly_rent_amount",
    *FIELD_NAMES,
]


def _csv_amount(value) -> str:
    """65000.0 reads as noise in a spreadsheet; 65000 reads as a rupee figure."""
    if value is None:
        return ""
    amount = float(value)
    return str(int(amount)) if amount.is_integer() else str(amount)


def export_csv(path: Optional[str] = None) -> str:
    """One row per lease, one column per field. This is the ops spreadsheet.

    `monthly_rent_amount` sits next to the verbatim `monthly_rent` so a reviewer
    can see what was summed without losing the original text. Failed attempts are
    included, with their `error`, so the register shows what still needs work
    rather than looking quietly complete.
    """
    rows = list_leases(path, limit=1_000_000)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()

    for row in rows:
        detail = get_lease(row["id"], path) or {}
        flat = {k: row.get(k) for k in CSV_COLUMNS}
        flat["needs_review"] = "yes" if detail.get("needs_review") else "no"
        flat["monthly_rent_amount"] = _csv_amount(row.get("monthly_rent_amount"))
        for name in FIELD_NAMES:
            flat[name] = detail.get("fields", {}).get(name) or ""
        writer.writerow(flat)

    return buffer.getvalue()
