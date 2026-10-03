"""
HTTP API around the lease extractor.

The Groq key and the LLM live here. Streamlit (and any other client) only sends
text or a PDF and receives validated JSON plus per-field evidence -- it never
talks to Groq.

Two layers, deliberately kept apart:

  /extract*   stateless. Runs an extraction and returns it. Writes nothing.
              Useful for one-off calls and for tests.
  /leases*    the register. Extracts and remembers, so results accumulate.

Keeping /extract side-effect free means a caller can retry a single document
without risking duplicate rows, and it is why the extraction tests need no
database at all.

Status codes are chosen so a client can react without parsing prose:
  400 malformed input        401 bad/missing token
  413 input too large       422 unreadable PDF or unusable model output
  502 model provider failed  503 server missing GROQ_API_KEY
"""

import functools
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import registry
from documents import DocumentTooLarge, text_from_pdf_bytes
from extractor import FIELD_NAMES, LeaseDetails, extract_lease_details
from llm_utils import LLMCallError

load_dotenv()

MAX_TEXT_CHARS = 200_000
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(
    title="Lease Document Extractor",
    description=(
        "Structured extraction from a single lease, with the source sentence for "
        "each value, saved to a local lease register."
    ),
    version="3.0.0",
)


class ExtractRequest(BaseModel):
    # min_length only; the upper bound is enforced in the endpoint so we can
    # answer 413 instead of letting pydantic answer a generic 422.
    text: str = Field(..., min_length=1)


class SaveLeaseRequest(ExtractRequest):
    source_name: str = Field(
        "pasted text", max_length=300, description="Filename or label for the register"
    )
    source_kind: str = Field("text", pattern="^(pdf|text)$")


class ExtractionResponse(BaseModel):
    """Explicit envelope. v1 returned a bare nine-key object; nesting the values
    under `fields` leaves room for evidence and warnings without breaking the
    meaning of the top level."""

    fields: LeaseDetails
    evidence: Dict[str, Optional[str]] = Field(
        description="Field name -> verbatim source sentence, or null"
    )
    missing_fields: List[str] = Field(
        default_factory=list, description="Fields absent from the document (null)"
    )
    unverified_fields: List[str] = Field(
        default_factory=list,
        description=(
            "Values returned without a quote that verifiably supports them. "
            "Review these before trusting the extraction."
        ),
    )
    unexpected_fields: List[str] = Field(
        default_factory=list,
        description="Non-schema keys the model returned; surfaced, not silently dropped",
    )
    no_fields_found: bool = Field(
        default=False,
        description=(
            "True when every field is null. Usually an extraction failure rather "
            "than a genuinely empty document."
        ),
    )
    degraded: bool = Field(
        default=False,
        description="True if constrained decoding was unavailable and JSON was salvaged",
    )


def _missing_key() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="GROQ_API_KEY is not set on the API server.",
    )


# openai/gpt-oss-120b is the default because it is the smallest hosted model
# that actually honours strict Structured Outputs on Groq. gpt-oss-20b accepts
# the request and then returns an empty completion, which fails validation; the
# fallback chain in extractor.py recovers from that, but only the strict path
# guarantees all nine rows come back.
DEFAULT_MODEL = "openai/gpt-oss-120b"


@functools.lru_cache(maxsize=1)
def _cached_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY", "").strip(),
        max_retries=0,  # retries are llm_utils' job, not the SDK's
    )


def get_llm():
    if not os.getenv("GROQ_API_KEY", "").strip():
        raise _missing_key()
    return _cached_llm()


def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("EXTRACTOR_API_TOKEN", "").strip()
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")


def _to_response(result) -> ExtractionResponse:
    return ExtractionResponse(
        fields=result.details,
        evidence=result.evidence,
        missing_fields=result.missing,
        unverified_fields=result.unverified_fields,
        unexpected_fields=result.unexpected_fields,
        no_fields_found=result.no_fields_found,
        degraded=result.degraded,
    )


def run_extraction(text: str, llm) -> ExtractionResponse:
    return _to_response(extract_or_fail(text, llm))


def extract_or_fail(text: str, llm):
    """Run the extraction, translating failures into status codes.

    Split out from run_extraction so the saving endpoints can persist the raw
    result (which carries evidence and the degraded flag) instead of only the
    response envelope.
    """
    try:
        return extract_lease_details(llm, text)
    except LLMCallError as err:
        raise HTTPException(status_code=502, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err


@app.get("/health")
def health():
    """Liveness: is the process up. Says nothing about whether it can work."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: can this process actually serve an extraction request?"""
    if not os.getenv("GROQ_API_KEY", "").strip():
        return {"status": "not_ready", "reason": "GROQ_API_KEY is not set"}
    return {"status": "ready", "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL)}


def _enforce_text_limit(text: str) -> str:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"text is {len(text):,} characters; the limit is {MAX_TEXT_CHARS:,}. "
                "Send the relevant pages only."
            ),
        )
    return text


def _read_pdf(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf uploads are supported. Paste text via POST /extract.",
        )

    # Read one byte past the cap so an oversize body is detected without ever
    # buffering all of it.
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
                "Split the lease or paste the text."
            ),
        )

    try:
        text = text_from_pdf_bytes(raw)
    except DocumentTooLarge as err:
        raise HTTPException(status_code=413, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    return _enforce_text_limit(text)


@app.post("/extract")
def extract(req: ExtractRequest, _: None = Depends(require_token)):
    text = _enforce_text_limit(req.text)
    return run_extraction(text, get_llm())


@app.post("/extract/file")
def extract_file(file: UploadFile = File(...), _: None = Depends(require_token)):
    return run_extraction(_read_pdf(file), get_llm())


# --------------------------------------------------------------------------
# The register
# --------------------------------------------------------------------------


class SavedLeaseResponse(BaseModel):
    """An extraction plus the register row it produced."""

    lease_id: int
    created: bool = Field(
        description="False when this document was already registered and refreshed"
    )
    needs_review: bool
    monthly_rent_amount: Optional[float] = Field(
        default=None,
        description="Parsed rent for summing. Null when the text was not "
        "confidently a number; the verbatim string is always in `fields`.",
    )
    extraction: ExtractionResponse


@app.post("/leases", status_code=201)
def save_lease(req: SaveLeaseRequest, _: None = Depends(require_token)):
    """Extract and remember. Re-posting the same document refreshes its row."""
    text = _enforce_text_limit(req.text)
    result = extract_or_fail(text, get_llm())
    saved = registry.save_extraction(
        text=text,
        source_name=req.source_name,
        source_kind=req.source_kind,
        extraction=result,
    )
    return SavedLeaseResponse(
        lease_id=saved["lease_id"],
        created=saved["created"],
        needs_review=saved["needs_review"],
        monthly_rent_amount=saved["monthly_rent_amount"],
        extraction=_to_response(result),
    )


@app.post("/leases/file", status_code=201)
def save_lease_file(file: UploadFile = File(...), _: None = Depends(require_token)):
    text = _read_pdf(file)
    result = extract_or_fail(text, get_llm())
    saved = registry.save_extraction(
        text=text,
        source_name=file.filename or "uploaded.pdf",
        source_kind="pdf",
        extraction=result,
    )
    return SavedLeaseResponse(
        lease_id=saved["lease_id"],
        created=saved["created"],
        needs_review=saved["needs_review"],
        monthly_rent_amount=saved["monthly_rent_amount"],
        extraction=_to_response(result),
    )


# Literal paths are declared before /leases/{lease_id} so they are not captured
# by the int path parameter.
@app.get("/leases.csv", response_class=PlainTextResponse)
def leases_csv(_: None = Depends(require_token)):
    return PlainTextResponse(
        content=registry.export_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="lease_register.csv"'},
    )


@app.get("/leases")
def leases(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: str = Query("", max_length=200),
    _: None = Depends(require_token),
):
    return {
        "total": registry.count_leases(search=search),
        "leases": registry.list_leases(limit=limit, offset=offset, search=search),
    }


@app.get("/leases/{lease_id}")
def lease_detail(lease_id: int, _: None = Depends(require_token)):
    record = registry.get_lease(lease_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No lease with id {lease_id}")
    return record


@app.delete("/leases/{lease_id}")
def remove_lease(lease_id: int, _: None = Depends(require_token)):
    if not registry.delete_lease(lease_id):
        raise HTTPException(status_code=404, detail=f"No lease with id {lease_id}")
    return {"deleted": lease_id}


__all__ = ["app", "ExtractionResponse", "SavedLeaseResponse", "FIELD_NAMES"]
