# Lease Register

Interview-sized slice of a real-estate ops tool: turn unstructured leases into
**validated JSON** (tenant, landlord, address, term, rent, deposit, renewal)
**plus the sentence each value came from** — and keep the results in a **local
register** that grows into a rent roll.

The UI is a client. The model and API key live on a small FastAPI service.

```
Streamlit  --HTTP-->  FastAPI  -->  Groq  -->  Pydantic LeaseDetails
   app.py              api.py     extractor.py        |
     |                              |                 v
     +-------- register tab ------> registry.py --> SQLite (lease_register.db)
```

Missing fields come back as `null`. The model is told not to guess, and every
value is re-checked against the source text before being reported as verified.

## What changed in v3

v1 asked the model for JSON in a prompt and hoped for the best. v2 made the model
**structurally unable** to return the wrong shape, and made every value
**checkable**. v3 makes the result **keep** — persistence, dedupe, search, and CSV.

| Area | v1 | v2 | v3 |
|---|---|---|---|
| Output guarantee | Prompt + JSON parsing | strict Structured Outputs | unchanged |
| Provenance | none | verbatim quote per field | stored with the row |
| Trust | blind | unverified values flagged | flag persists, shown in the register |
| **Storage** | **none** | **none** | **SQLite register, re-runnable without duplicates** |
| **Rent as a number** | **none** | **none** | **parsed alongside the verbatim string** |
| **Register view** | **none** | **none** | **search, per-lease evidence, delete** |
| **CSV** | **none** | **reviewed JSON only** | **one row per lease, every field** |
| Failed attempts | invisible | 502 and gone | recorded with the reason |
| Decode strategy | one path, no fallback | chain existed but only ran if a strategy failed to **build** | chain runs when the provider **rejects** a call |
| Default model | — | `gpt-oss-20b` (returns empty under strict) | `gpt-oss-120b` (measured 9/9 rows) |
| Tests | 16 | 104 | 224 |

## Two layers, deliberately separate

| Route | Writes? | Use for |
|---|---|---|
| `/extract`, `/extract/file` | **no** | one-off calls, tests, safe retries |
| `/leases`, `/leases/file` | yes | the register |

`/extract` is stateless on purpose: retrying a single document must never risk a
duplicate row, and it is why the extraction tests need no database at all.

## The register

`registry.py` — stdlib `sqlite3`, two tables:

- `leases` — one row per document: content hash, filename, status, timestamps,
  counts, and the parsed rent number.
- `lease_fields` — one row per field per lease: value, source quote, verified flag.

Decisions worth knowing:

- **Dedupe by document content, not filename.** The key is a SHA-256 of the
  text with whitespace folded, because PDF text extraction is not byte-stable and
  a document that renders with different line breaks is still the same lease.
  Re-running the same lease **refreshes its row** rather than adding a second one,
  which is what makes a corrected extraction able to replace a bad earlier one.
- **Failures are rows too.** An unreadable PDF is recorded with its reason, so
  the register shows what still needs work instead of looking quietly complete.
- **Fields are always all nine.** A sparse lease returns `null`, never a missing
  key, so callers never have to guard.
- **Rent is parsed, but the original is kept.** `monthly_rent` stays exactly as
  written (`"Rs. 65,000"`); `monthly_rent_amount` is a separate best-effort
  `65000.0` for summing. Parsing is ambiguous-means-`None` by design
  (`amounts.py`): `"11 months"` is never read as 11.
- **Unverified survives the round trip.** A value the model could not back with
  a real sentence stays flagged, and the row is marked `needs_review`.
- **A fresh connection per operation**, so the service stays correct under
  Streamlit's threaded requests.

## Trust, not just extraction

Every non-null value must be backed by a `source_quote` that passes two
independent checks:

1. the quote appears in the source document, and
2. the value appears inside the quote.

Either alone gives false alarms — (1) alone accepts a quote about the deposit
paired with a rent value; (2) alone accepts a value stitched from two clauses.
Values that fail are returned but listed in `unverified_fields`, and the UI
marks them in red. This costs no extra model call, and the flag is stored, so a
lease that needed a human still looks like it does next week.

## The core idea

`openai/gpt-oss-120b` (the default) supports Groq Structured Outputs with
`strict: true`, which uses **grammar-constrained decoding**. The model cannot
emit a field name outside the enum, a non-nullable value, or invalid JSON.

Strict mode normally requires every field to be `required` with
`additionalProperties: false`. That looks like it would forbid nulls — it does
not. The generated schema marks each field **required but nullable**
(`anyOf: [string, null]`), so "absent from the document" is still expressible.
That is what keeps the no-guessing promise intact under a hard schema guarantee,
and `test_structured_output.py` locks the property in.

The model returns one row per field rather than a flat object, because an
open-ended evidence dict cannot satisfy `additionalProperties: false`, but a list
of fixed-shape rows with an enum `field` column can.

```json
{"fields": [
  {"field": "monthly_rent", "value": "Rs. 65,000", "source_quote": "monthly rent of Rs. 65,000"},
  {"field": "landlord_name", "value": null, "source_quote": null}
]}
```

### Why 120b and not 20b

Measured against the live API, not assumed:

| Model | strict | rows returned | quotes returned |
|---|---|---|---|
| `openai/gpt-oss-120b` | works | 9/9 | 9/9 |
| `openai/gpt-oss-20b` | **rejected at generation** | 4/9 | partial |

`gpt-oss-20b` *accepts* the strict request and then returns an empty
completion, which Groq rejects with `400 json_validate_failed`. The reply you
get back has fewer rows than the schema demands, so evidence goes missing and
every value ends up flagged unverified. On 120b the same lease comes back
`degraded: false`, `unverified: []`, all nine fields present.

### The fallback chain

Because building a strategy proves nothing — a provider can accept the request
and reject it at generation time — the chain is walked at **invoke** time:

```
json_schema(strict) → json_schema → json_mode → function_calling → prose+salvage
```

- A **400/422** means the provider refused *this decoding strategy*, so the next
  one is tried. The reply is marked `degraded`.
- A **401/403** (bad key) and **404** (wrong model) are not retried down the
  chain: they will fail identically four times. They surface immediately.
- **429 and 5xx** are retried with backoff by `llm_utils` before any of this.

`test_fallback.py` covers each of those branches offline.

## Run locally

Two terminals. Groq key only on the API.

```bash
pip install -r requirements.txt
# cp .env.example .env, then set:
# GROQ_API_KEY=your_key_here          # console.groq.com
# EXTRACTOR_API_URL=http://127.0.0.1:8001
# EXTRACTOR_API_TOKEN=                 # optional; if set, UI and API must match
# REGISTRY_DB=lease_register.db        # where the register is written

uvicorn api:app --reload --port 8001
streamlit run app.py
```

```bash
pytest              # offline: fake LLM + TestClient, no API key, temp DBs
pytest --cov        # if pytest-cov is installed
```

Check the API can actually reach the model before blaming the UI:

```bash
curl http://127.0.0.1:8001/ready
# {"status":"ready","model":"openai/gpt-oss-120b"}   <- good
# {"status":"not_ready","reason":"GROQ_API_KEY is not set"}   <- the key is missing
```

Demo path: extract `lease_residential_01` (complete, every field verifiable) with
**Save to the lease register** ticked, then `lease_incomplete_01` (nulls, not an
invented rent). Switch to the **Register** tab: two rows, search, open one to see
its evidence, download the CSV. Re-run the first sample to watch it refresh
rather than duplicate.

### If `/ready` says not_ready

`load_dotenv()` runs **once, when the API process starts**. `--reload` watches
`.py` files, not `.env`. So if you create or edit `.env` after starting the API,
nothing changes until you restart it — and you will keep seeing the old error.

## Layout

- **`registry.py`** — SQLite register: schema, content-hash upsert, queries, CSV.
- **`amounts.py`** — best-effort Indian-format amount parsing; ambiguity → `None`.
- **`extractor.py`** — schema, prompt, structured-output negotiation, JSON salvage, provenance checks.
- **`llm_utils.py`** — retry transient errors; fail fast on 400/401/403/404/422.
- **`api.py`** — `/extract*` (stateless) and `/leases*` (the register).
- **`documents.py`** — PDF text layer only; byte/page/char caps.
- **`app.py`** — Streamlit: Extract tab, Register tab, review grid, CSV download.
- **`sample_leases/`** — synthetic residential, commercial, incomplete leases (`.txt` + `.pdf`).

## Tests

209 tests, none needing a key or a network call.

| File | Covers |
|---|---|
| `test_extractor.py` | schema handling, no-guessing, provenance, JSON salvage |
| `test_structured_output.py` | the exact wire schema sent to Groq; fallback chain |
| `test_llm_utils.py` | backoff, non-retryable statuses, nested httpx errors |
| `test_api.py` | status codes, size limits, auth, response shape |
| `test_register_api.py` | `/leases*`, dedupe, search, CSV, auth, `/extract` stays stateless |
| `test_registry.py` | schema, upsert, orphans, cascade delete, CSV escaping |
| `test_amounts.py` | lakh/crore, ambiguity → `None`, Indian digit grouping |
| `test_fallback.py` | decode-strategy fallback at invoke time; which statuses justify it |
| `test_documents.py` | PDF text layer, caps, encrypted and image-only files |
| `test_integration.py` | real sample leases end to end, with a scripted model |

`test_structured_output.py` needs no key and makes no network call — it
introspects the request langchain-groq builds. If a dependency upgrade changes
that shape, the suite fails instead of the promise breaking quietly.

## HTTP

`POST /extract` `{"text": "..."}` — stateless.

```json
{
  "fields":         {"tenant_name": "Priya Nair", "monthly_rent": "Rs. 65,000", "...": null},
  "evidence":       {"monthly_rent": "monthly rent of Rs. 65,000"},
  "missing_fields": ["landlord_name"],
  "unverified_fields": [],
  "unexpected_fields": [],
  "no_fields_found": false,
  "degraded": false
}
```

The register:

| Route | Does |
|---|---|
| `POST /leases` | extract + save; `source_name`, `source_kind` (`text`/`pdf`) |
| `POST /leases/file` | same, from a text-layer PDF |
| `GET /leases` | `search`, `limit`, `offset`; `{total, leases}` |
| `GET /leases.csv` | every lease, every field, as a download |
| `GET /leases/{id}` | one lease with fields, evidence, unverified list |
| `DELETE /leases/{id}` | remove it and its fields |

`POST /leases` answers 201 with the id and the parsed rent next to the full
extraction, so the client does not need a second request:

```json
{"lease_id": 1, "created": true, "needs_review": false,
 "monthly_rent_amount": 65000.0, "extraction": {"...": "same shape as /extract"}}
```

| Code | Meaning |
|---|---|
| 400 | malformed input (blank text, non-PDF upload) |
| 401 | bad or missing `EXTRACTOR_API_TOKEN` |
| 404 | no lease with that id |
| 413 | input too large (text, PDF bytes, pages, or extracted text) |
| 422 | unreadable PDF, or model output that cannot be used |
| 502 | model provider failed after retries |
| 503 | `GROQ_API_KEY` not set on the API |

## Honest limits

- **`lease_register.db` holds real lease data** — names, addresses, rents. It is
  gitignored, but a local SQLite file is not access control. Put it somewhere
  deliberate, and encrypt the disk.
- **No OCR for scans.** No legal or compliance check. No batch upload, no job
  queue, no user accounts.
- `EXTRACTOR_API_TOKEN` is a shared secret, not user login. Comparison is not
  constant-time. Anyone who can reach the API can delete rows.
- Extracted amounts and dates stay **as written** (strings). A wrong type from the
  model is a 422, not a coerce. The numeric rent is derived and may be `null`.
- `unverified_fields` catches quotes that do not support their value. It cannot
  catch a value that is genuinely in the document but read with the wrong
  meaning — that still needs a human.
- Verification is substring-based, so a paraphrased-but-correct quote is flagged
  as unverified. False positives are the intended failure direction.
- `needs_review` is also set when fields are simply missing. A sparse lease is a
  normal outcome, not necessarily a mistake.

## If this were a product

Next: OCR for scanned leases, batch upload with per-document status, a real
review queue with human labels feeding back into field-level confidence, and
tenant/landlord entity resolution across a portfolio. The extraction is the easy
part; the review loop is the product.