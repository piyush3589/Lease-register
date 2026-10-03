"""
Streamlit client. All extraction happens on the FastAPI process -- this file
never holds GROQ_API_KEY and never imports the extractor or the registry.

Two tabs, because there are two jobs:

  Extract   one lease in, fields and evidence out, correct and export.
  Register  everything extracted so far, searchable, downloadable as CSV.

The register is the part that makes this a tool rather than a demo: results
persist, so the second extraction of the week builds on the first.
"""

import glob
import json
import os
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("EXTRACTOR_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.getenv("EXTRACTOR_API_TOKEN", "").strip()

# Mirrors the API's cap so the UI can warn before spending a round trip.
MAX_TEXT_CHARS = 200_000

st.set_page_config(page_title="Lease Document Extractor", layout="wide")
st.title("Lease Document Extractor")
st.caption(
    "Pulls structured fields (tenant, rent, term, deposit, renewal) out of raw "
    "lease text, along with the sentence each value came from. Missing fields come "
    "back as null, not a guess. Everything is kept in a local lease register so the "
    "rent roll builds up over time."
)

# Status codes the UI explains differently, rather than dumping a raw detail string.
ERROR_HELP = {
    400: "The request was rejected as malformed.",
    401: "The API token is missing or wrong. Check EXTRACTOR_API_TOKEN on both sides.",
    404: "That lease is no longer in the register.",
    413: "The document is too large. Send only the relevant pages.",
    422: "The document or the model's output could not be processed.",
    502: "The model provider failed. This is usually transient -- try again.",
    503: "The API server has no GROQ_API_KEY set.",
}


def api_headers() -> dict:
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def _report_failure(response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    help_text = ERROR_HELP.get(response.status_code, "")
    st.error(f"Request failed ({response.status_code}). {help_text}\n\n{detail}")


def _request(method: str, path: str, timeout: int = 120, **kwargs):
    """Returns the Response, or None after showing the error. Never raises."""
    try:
        response = requests.request(
            method, f"{API_URL}{path}", headers=api_headers(), timeout=timeout, **kwargs
        )
    except requests.RequestException as err:
        st.error(
            f"Could not reach the extractor API at {API_URL}. "
            f"Start it with `uvicorn api:app --reload --port 8000`. ({err})"
        )
        return None

    if response.status_code >= 400:
        _report_failure(response)
        return None
    return response


def extract_text(text: str, save: bool, source_name: str) -> Optional[dict]:
    path = "/leases" if save else "/extract"
    payload = {"text": text}
    if save:
        payload["source_name"] = source_name
    response = _request("POST", path, json=payload)
    return response.json() if response is not None else None


def extract_pdf(filename: str, raw: bytes, save: bool) -> Optional[dict]:
    path = "/leases/file" if save else "/extract/file"
    response = _request("POST", path, files={"file": (filename, raw, "application/pdf")})
    return response.json() if response is not None else None


def label(field: str) -> str:
    return field.replace("_", " ").title()


# --------------------------------------------------------------------------
# Rendering shared by both tabs
# --------------------------------------------------------------------------


def show_diagnostics(data: dict) -> None:
    """Surface what makes an extraction untrustworthy, before the values, so it
    is not read past."""
    if data.get("no_fields_found"):
        st.error(
            "**The model found nothing at all.** That usually means the text was "
            "not a readable lease -- not that the lease lacked these fields. "
            "Check the document, then retry."
        )

    unverified = data.get("unverified_fields") or []
    if unverified:
        st.warning(
            f"**{len(unverified)} value(s) could not be verified against the "
            f"document:** {', '.join(label(f) for f in unverified)}. For each one "
            "the model returned a value but no quotable sentence supporting it. "
            "Treat these as claims, not facts."
        )

    if data.get("degraded"):
        st.info(
            "Constrained decoding was unavailable for this model, so the JSON "
            "reply was salvaged after the fact. Results are less reliable than the "
            "verified path."
        )

    if data.get("unexpected_fields"):
        st.info(
            f"The model also returned keys outside the schema "
            f"({', '.join(data['unexpected_fields'])}). They are shown in the raw "
            "JSON but were not treated as lease fields."
        )

    missing = data.get("missing_fields") or []
    if missing:
        st.caption(
            f"{len(missing)} field(s) not stated in this document: "
            f"{', '.join(label(f) for f in missing)}. Expected for informal or "
            "incomplete leases -- the model is instructed not to guess."
        )

    found = sum(1 for v in data["fields"].values() if v is not None)
    st.caption(f"{found} of {len(data['fields'])} fields extracted.")


def show_fields(fields: dict, evidence: dict, unverified: list) -> None:
    unverified_set = set(unverified or [])
    for field, value in fields.items():
        name = label(field)
        if value is None:
            st.markdown(f"**{name}:** :orange[not found in document]")
            continue
        suffix = "  :red[unverified]" if field in unverified_set else ""
        st.markdown(f"**{name}:** {value}{suffix}")
        quote = (evidence or {}).get(field)
        if quote:
            st.caption(f"> {quote}")
        elif field in unverified_set:
            st.caption(":red[No supporting sentence was returned for this value.]")


def review_and_export(fields: dict) -> None:
    """Let a human correct values and take the result away.

    An LLM will be wrong sometimes. Making correction cheap is what makes the
    output usable downstream.
    """
    st.subheader("Review and export")
    st.caption(
        "Correct anything that looks wrong. Blank means the field is genuinely "
        "absent from the document."
    )

    edited = st.data_editor(
        [{"field": f, "value": v if v is not None else ""} for f, v in fields.items()],
        column_config={
            "field": st.column_config.TextColumn("Field", disabled=True, width="medium"),
            "value": st.column_config.TextColumn("Value", width="large"),
        },
        hide_index=True,
        use_container_width=True,
        key="review_grid",
    )

    reviewed = {row["field"]: (row["value"].strip() or None) for row in edited}
    corrected = [f for f, v in reviewed.items() if v != fields.get(f)]
    if corrected:
        st.info(f"Corrected {len(corrected)} field(s): {', '.join(label(f) for f in corrected)}")

    st.download_button(
        "Download reviewed JSON",
        data=json.dumps(reviewed, indent=2, ensure_ascii=False),
        file_name="lease_fields.json",
        mime="application/json",
    )


# --------------------------------------------------------------------------
# Tab 1: extract
# --------------------------------------------------------------------------


def extract_tab() -> None:
    sample_files = sorted(glob.glob("sample_leases/*.txt"))
    sample_names = [os.path.basename(f) for f in sample_files]

    col_input, col_help = st.columns([2, 1])

    with col_input:
        choice = st.selectbox(
            "Load a sample lease, or paste your own below",
            ["(paste my own)"] + sample_names,
        )
        uploaded = st.file_uploader("Or upload a text-layer PDF", type=["pdf"])

    with col_help:
        st.markdown(
            "**Try both samples.** `lease_residential_01` is complete. "
            "`lease_incomplete_01` has no landlord or dates -- those come back null "
            "rather than invented."
        )

    if uploaded is None:
        if choice != "(paste my own)":
            with open(os.path.join("sample_leases", choice), encoding="utf-8") as f:
                default_text = f.read()
        else:
            default_text = ""
        text = st.text_area("Lease document text", value=default_text, height=280)
        if len(text) > MAX_TEXT_CHARS:
            st.warning(
                f"{len(text):,} characters pasted; the limit is {MAX_TEXT_CHARS:,}. "
                "Trim it before extracting."
            )
        source_name = choice if choice != "(paste my own)" else "pasted text"
        payload = None
    else:
        payload = uploaded.getvalue()
        source_name = uploaded.name
        st.info(f"Will extract from uploaded file: {source_name} ({len(payload) / 1024:.0f} KB)")

    save = st.checkbox(
        "Save to the lease register",
        value=True,
        help="Re-running the same document refreshes its existing row rather than "
        "adding a duplicate.",
    )

    if st.button("Extract fields", type="primary"):
        with st.spinner("Extracting..."):
            if uploaded is not None:
                data = extract_pdf(uploaded.name, payload, save)
            elif text.strip():
                data = extract_text(text, save, source_name)
            else:
                st.warning("Paste lease text or upload a PDF first.")
                data = None

        if not data:
            return

        extraction = data.get("extraction", data)

        if data.get("lease_id"):
            if data["created"]:
                st.success(f"Saved to the register as lease #{data['lease_id']}.")
            else:
                st.info(
                    f"Already in the register as lease #{data['lease_id']} -- "
                    "refreshed it rather than adding a duplicate."
                )
            amount = data.get("monthly_rent_amount")
            st.caption(
                f"Parsed rent for summing: Rs. {amount:,.0f}"
                if amount is not None
                else "Rent could not be parsed into a number; the verbatim text is "
                "still in the register."
            )

        show_diagnostics(extraction)
        st.subheader("Extracted fields")
        show_fields(extraction["fields"], extraction.get("evidence"),
                    extraction.get("unverified_fields"))
        with st.expander("Raw JSON"):
            st.json(extraction)
        review_and_export(extraction["fields"])


# --------------------------------------------------------------------------
# Tab 2: register
# --------------------------------------------------------------------------


def register_tab() -> None:
    st.subheader("Lease register")
    st.caption(
        "Everything extracted so far. Same document re-run refreshes its row "
        "instead of adding another."
    )

    col_search, col_download = st.columns([3, 1])
    with col_search:
        search = st.text_input("Search by filename, tenant, or any field value", key="reg_search")
    with col_download:
        st.write("")
        response = _request("GET", "/leases.csv", timeout=60)
        if response is not None:
            st.download_button(
                "Download all as CSV",
                data=response.text,
                file_name="lease_register.csv",
                mime="text/csv",
                key="dl_all_csv",
            )

    listing = _request("GET", "/leases", params={"search": search, "limit": 500}, timeout=60)
    if listing is None:
        return

    body = listing.json()
    rows = body["leases"]
    st.caption(f"{body['total']} lease(s) in the register; showing {len(rows)}.")

    if not rows:
        st.info("Nothing here yet. Extract a lease in the first tab.")
        return

    needs_review = [r for r in rows if r["status"] != "ok" or r["unverified_count"] or r["no_fields_found"]]
    if needs_review:
        st.warning(
            f"{len(needs_review)} lease(s) need a human look: failed, nothing found, "
            "or a value with no supporting sentence."
        )

    table = [
        {
            "id": r["id"],
            "source": r["source_name"],
            "tenant": r["tenant_name"] or "",
            "rent": r["monthly_rent"] or "",
            "rent (num)": r["monthly_rent_amount"],
            "status": r["status"],
            "verified": f"{r['fields_found'] - r['unverified_count']}/{r['fields_found']}",
            "updated": (r["updated_at"] or "")[:16].replace("T", " "),
        }
        for r in rows
    ]
    st.dataframe(table, hide_index=True, use_container_width=True)

    ids = [r["id"] for r in rows]
    chosen = st.selectbox(
        "Open a lease to see its evidence", ids, format_func=lambda i: f"#{i}"
    )

    detail_response = _request("GET", f"/leases/{chosen}", timeout=60)
    if detail_response is None:
        return
    detail = detail_response.json()

    if detail["status"] != "ok":
        st.error(f"This lease failed to extract: {detail.get('error')}")
        return

    st.markdown(f"**{detail['source_name']}** · updated {detail['updated_at']}")
    if detail["needs_review"]:
        st.warning("This record is flagged for review.")

    show_fields(detail["fields"], detail["evidence"], detail["unverified_fields"])

    with st.expander("Raw JSON"):
        st.json(detail)

    if st.button("Delete this lease", key="delete_lease"):
        deleted = _request("DELETE", f"/leases/{chosen}", timeout=60)
        if deleted is not None:
            st.success(f"Deleted lease #{chosen}.")
            st.rerun()


extract_pane, register_pane = st.tabs(["Extract a lease", "Register"])
with extract_pane:
    extract_tab()
with register_pane:
    register_tab()
