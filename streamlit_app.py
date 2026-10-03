"""
Streamlit app (standalone) - runs entirely on Streamlit Cloud without a separate FastAPI server.
All extraction, verification, and registry logic runs directly in-process.

Two tabs:
  Extract   one lease in, fields and evidence out, correct and export.
  Register  everything extracted so far, searchable, downloadable as CSV.
"""

import io
import json
from typing import Optional

import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Local modules
import documents
import extractor
import llm_utils
import registry

load_dotenv()

# Mirrors API caps so UI can warn early
MAX_TEXT_CHARS = 200_000
MAX_PDF_BYTES = 15 * 1024 * 1024  # 15MB
MAX_PAGES = 60

st.set_page_config(page_title="Lease Document Extractor", layout="wide")
st.title("Lease Document Extractor")
st.caption(
    "Pulls structured fields (tenant, rent, term, deposit, renewal) out of raw "
    "lease text, along with the sentence each value came from. Missing fields come "
    "back as null, not a guess. Everything is kept in a local lease register so the "
    "rent roll builds up over time."
)


@st.cache_resource
def get_llm():
    """Initialize and cache LLM client using Streamlit secrets (preferred) or env vars."""
    # Try Streamlit secrets first (Streamlit Cloud)
    api_key = None
    model = "openai/gpt-oss-120b"
    
    try:
        if hasattr(st, "secrets"):
            secrets = st.secrets
            api_key = secrets.get("GROQ_API_KEY") or secrets.get("gsk", None)
            model = secrets.get("GROQ_MODEL", model) or model
    except Exception:
        pass
    
    # Fallback to environment variables
    if not api_key:
        import os
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model = os.getenv("GROQ_MODEL", model).strip() or model
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to Streamlit secrets (GROQ_API_KEY) or environment variables."
        )
    
    return ChatGroq(model=model, api_key=api_key)


def extract_text(text: str, save: bool, source_name: str) -> Optional[dict]:
    """Extract from plain text. Returns dict matching API response shape."""
    try:
        llm = get_llm()
    except Exception as err:
        st.error(f"Failed to initialize LLM: {err}")
        return None

    if len(text) > MAX_TEXT_CHARS:
        st.error(f"Text exceeds {MAX_TEXT_CHARS:,} characters ({len(text):,}).")
        return None

    try:
        res = extractor.extract_lease_details(llm, text)
        data = res.to_dict()
        
        if save:
            try:
                saved = registry.save_extraction(text, source_name, res)
                data["saved"] = True
                data["lease_id"] = saved.id
                data["created"] = saved.created
            except Exception as e:
                st.warning(f"Extraction succeeded but saving to register failed: {e}")
                data["saved"] = False
                data["save_error"] = str(e)
        else:
            data["saved"] = False
        return data
    except llm_utils.LLMCallError as e:
        st.error(f"LLM call failed: {e}")
        return None
    except ValueError as e:
        st.error(f"Invalid document or model output: {e}")
        return None
    except Exception as e:
        st.error(f"Extraction failed: {e}")
        return None


def extract_pdf(filename: str, raw: bytes, save: bool) -> Optional[dict]:
    """Extract from PDF bytes."""
    try:
        # Parse PDF
        text, errors = documents.text_from_pdf_bytes(raw, max_pages=MAX_PAGES, max_chars=MAX_TEXT_CHARS)
        if errors:
            for err in errors:
                st.error(err)
            return None
        if not text:
            st.error("PDF contains no extractable text (scanned/image-only PDF?).")
            return None
    except ValueError as e:
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return None

    return extract_text(text, save, filename)


def label(field: str) -> str:
    return field.replace("_", " ").title()


def show_diagnostics(data: dict) -> None:
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


def show_results(data: dict) -> None:
    if data is None:
        return
    show_diagnostics(data)

    if data.get("saved"):
        if data.get("created"):
            st.success(f"Saved to lease register (ID {data['lease_id']}).")
        else:
            st.info(f"Already in the register (ID {data['lease_id']}); refreshed it rather than adding a duplicate.")

    fields = data.get("fields", {})
    evidence = data.get("evidence", {})

    st.subheader("Extracted fields")
    grid = st.columns(2)
    order = [
        "tenant_name",
        "landlord_name",
        "property_address",
        "lease_start_date",
        "lease_end_date",
        "lease_term",
        "monthly_rent",
        "security_deposit",
        "renewal_terms",
    ]
    for i, f in enumerate(order):
        with grid[i % 2]:
            val = fields.get(f)
            ev = evidence.get(f)
            with st.container(border=True):
                st.markdown(f"**{label(f)}**")
                st.write(val if val is not None else "_Not found in document_")
                if ev:
                    st.caption(f"Source quote: “{ev}”")
                    if data.get("unverified_fields") and f in data["unverified_fields"]:
                        st.caption("⚠️ This quote could not be verified against the document.")
                    else:
                        st.caption("✓ Verified against document text.")

    with st.expander("Raw response (for debugging)"):
        st.json(data)


# Extract tab
tab_extract, tab_register = st.tabs(["Extract a lease", "Register"])

with tab_extract:
    st.markdown("### Upload or paste a lease document")
    mode = st.radio(
        "Input type",
        ["Sample leases", "Paste my own", "Upload PDF"],
        horizontal=True,
    )

    text_to_extract = ""
    source_name = "pasted.txt"

    if mode == "Sample leases":
        samples = sorted(glob.glob("sample_leases/*.txt"))
        if not samples:
            st.warning("No sample_leases folder found in current directory.")
        choice = st.selectbox("Pick a sample", ["(none)"] + samples)
        if choice != "(none)":
            try:
                with open(choice, encoding="utf-8") as f:
                    text_to_extract = f.read()
                source_name = choice
            except Exception as e:
                st.error(f"Could not read sample: {e}")

    elif mode == "Paste my own":
        text_to_extract = st.text_area(
            "Paste lease text here",
            height=300,
            placeholder="Paste the full lease (or the relevant pages) here...",
        )
        source_name = "pasted.txt"

    elif mode == "Upload PDF":
        pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if pdf_file is not None:
            pdf_bytes = pdf_file.getvalue()
            if len(pdf_bytes) > MAX_PDF_BYTES:
                st.error(f"PDF exceeds {MAX_PDF_BYTES // (1024*1024)}MB limit.")
            else:
                # Process on button click
                pass

    col1, col2 = st.columns([3, 1])
    with col2:
        save_to_register = st.checkbox("Save to the lease register", value=True)

    if st.button("Extract fields", type="primary"):
        if mode == "Upload PDF":
            if pdf_file is None:
                st.error("Please upload a PDF first.")
            else:
                with st.spinner("Extracting lease details..."):
                    res = extract_pdf(pdf_file.name, pdf_file.getvalue(), save_to_register)
                show_results(res)
        else:
            if not text_to_extract.strip():
                st.error("Please provide some lease text first.")
            else:
                if len(text_to_extract) > MAX_TEXT_CHARS:
                    st.warning(
                        f"Text is {len(text_to_extract):,} chars (over {MAX_TEXT_CHARS:,} cap). "
                        "Consider trimming to the operative clauses."
                    )
                with st.spinner("Extracting lease details..."):
                    res = extract_text(text_to_extract, save_to_register, source_name)
                show_results(res)


# Register tab
with tab_register:
    st.markdown("### Lease register")
    try:
        leases = registry.list_leases()
    except Exception as e:
        st.error(f"Failed to load register: {e}")
        leases = []

    if not leases:
        st.info("No leases saved yet. Extract something with 'Save to the lease register' ticked.")
    else:
        # Search/filter
        q = st.text_input("Search (filename, tenant, landlord, address, status)", "")
        filtered = []
        ql = q.lower()
        for l in leases:
            hay = " ".join(
                str(x or "")
                for x in [
                    l.source_name,
                    l.tenant_name,
                    l.landlord_name,
                    l.property_address,
                    l.status,
                    l.error or "",
                ]
            ).lower()
            if not ql or ql in hay:
                filtered.append(l)

        st.write(f"Showing {len(filtered)} of {len(leases)} leases")

        # Select lease
        options = {f"#{l.id} {l.source_name} — {l.tenant_name or 'Unknown'}": l for l in filtered}
        sel = st.selectbox("Select a lease to inspect", list(options.keys()))
        chosen = options[sel] if sel else None

        if chosen:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status", chosen.status)
                c2.metric("Fields found", chosen.fields_found or 0)
                c3.metric("Unverified", chosen.unverified_count or 0)
                c4.metric("Degraded", "yes" if chosen.degraded else "no")

                st.write(f"**Source:** {chosen.source_name}")
                st.write(f"**Created:** {chosen.created_at} | **Updated:** {chosen.updated_at}")
                if chosen.error:
                    st.error(chosen.error)

            # Load full lease with fields
            try:
                full = registry.get_lease(chosen.id)
            except Exception as e:
                st.error(f"Failed to load lease details: {e}")
                full = None

            if full:
                # Show fields
                st.subheader("Fields & evidence")
                fields_map = {f.field_name: f for f in full.fields}
                evidence_map = {f.field_name: f.source_quote for f in full.fields}
                unverified_set = {f.field_name for f in full.fields if not f.verified}

                grid = st.columns(2)
                order = [
                    "tenant_name",
                    "landlord_name",
                    "property_address",
                    "lease_start_date",
                    "lease_end_date",
                    "lease_term",
                    "monthly_rent",
                    "security_deposit",
                    "renewal_terms",
                ]
                for i, fname in enumerate(order):
                    with grid[i % 2]:
                        ff = fields_map.get(fname)
                        with st.container(border=True):
                            st.markdown(f"**{label(fname)}**")
                            val = ff.value if ff else None
                            st.write(val if val is not None else "_Not found in document_")
                            if ff and ff.source_quote:
                                st.caption(f"Source quote: “{ff.source_quote}”")
                                if ff.verified:
                                    st.caption("✓ Verified against document text.")
                                else:
                                    st.caption("⚠️ Not verified against document text.")

                # Actions
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Delete this lease", type="secondary"):
                        try:
                            registry.delete_lease(chosen.id)
                            st.success("Deleted.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
                with col_b:
                    # Refresh - re-extract with stored text
                    if st.button("Refresh from stored content", type="secondary"):
                        # Need stored text - but registry stores hash, not raw text in our current model
                        # Easier: user can re-upload; but let's skip or warn
                        st.info("Refresh requires original text. Re-upload the same document in Extract tab (it will refresh).")

        # CSV export
        st.divider()
        if st.button("Download all as CSV", type="secondary"):
            try:
                csv_bytes = registry.export_csv()
                st.download_button(
                    label="Download lease_register.csv",
                    data=csv_bytes,
                    file_name="lease_register.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"CSV export failed: {e}")
