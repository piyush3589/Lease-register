# Lease Register v3

> Turn messy lease PDFs/text into **validated, auditable lease data** — with field-level proof, strict schema enforcement, and a searchable local rent roll.

Lease Register extracts key fields from residential/commercial leases, forces every result to match a strict schema, and proves each value by linking it back to the **exact sentence** it came from. Extracted leases are persisted to a local SQLite register so you can search, inspect evidence, fix, and export a clean CSV anytime.

## 1) What It Does

Given a lease document (PDF or plain text), the system returns a structured record with:

- **Core fields** – Tenant, Landlord, Property Address, Lease Start Date, Lease End Date, Lease Term, Monthly Rent (as written), Deposit, Renewal Clause
- **Provenance** – A source_quote (verbatim sentence) for every non-null field
- **Verification** – A erified flag showing whether that quote actually exists in the cleaned source text
- **Normalised rent** – monthly_rent_amount (best-effort numeric) alongside monthly_rent (exact string)
- **Auditability** – Failures, incomplete leases and "needs review" cases are all recorded in the register

## 2) Architecture

`	ext
+-------------+   HTTP   +------------+   Groq API   +-------------+
¦  Streamlit   ¦---------?¦   FastAPI   ¦-------------?¦  gpt-oss-120b¦
¦   (app.py)   ¦          ¦   (api.py)  ¦              ¦  (extractor) ¦
+-------------+          +------------+              +-------------+
       ¦                        ¦                             ¦
       ¦                        ¦                             ?
       ¦                        ¦                    Pydantic v2 Schema
       ¦                        ¦                      (LeaseDetails)
       ¦                        ?
       ¦                +-------------+
       ¦                ¦  Registry    ¦
       ¦                ¦ (registry.py)¦
       ¦                +-------------+
       ¦                        ¦
       ?                        ?
   Register UI            SQLite DB
  (search/evidence/       (lease_register.db)
   delete/export)
`

## 3) Key Features

| Feature | Description |
|---|---|
| **Strict Structured Outputs** | Uses Pydantic BaseModel with Field constraints so the LLM *cannot* return an arbitrary shape. Invalid JSON is rejected and retried with a safe fallback path. |
| **Field-Level Provenance** | Every extracted field stores the exact source_quote the model used to justify it. |
| **Source Verification** | extractor.py normalises the lease text and checks each quote appears in it. If it doesn't, the field is marked unverified and the lease is flagged 
eeds_review. |
| **Dual API Flows** | /extract* routes are **stateless** (safe for retries/tests). /leases* routes **persist** to SQLite (power the register). |
| **Content-Hash Deduplication** | egistry.py hashes the normalised document text (SHA-256). Re-submitting the same lease **refreshes** the existing row instead of creating a duplicate. |
| **Failure Logging** | Unreadable PDFs, parsing failures or extraction errors are written as ailed rows with ailure_reason, so nothing silently disappears. |
| **Smart Rent Parsing** | mounts.py extracts a numeric monthly_rent_amount from strings like Rs. 65,000/- while deliberately ignoring ambiguous values ("11 months", "per annum", etc.). The verbatim monthly_rent is always preserved. |
| **Searchable Register** | Filter by filename, tenant, landlord, address, status or free-text. Inspect per-field evidence (value + quote + verified). Delete or refresh entries. |
| **CSV Export** | Export the entire register as a flat CSV with all 9 fields, rent amount, verification flags, status, filename, timestamps and content hash. |
| **Robust LLM Handling** | llm_utils.py implements a multi-strategy decode/fallback to handle provider edge-cases (including strict-output rejections) without corrupting the required schema. |
| **PDF + TXT Support** | documents.py reads .txt directly and extracts text from PDFs via pypdf. |
