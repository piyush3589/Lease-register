"""
documents.py
Turns uploaded files into plain text the extractor can consume.

Text-layer PDFs only. Scanned pages have no characters to extract, and this
project does not include OCR (a real product would need it).
"""

from io import BytesIO

from pypdf import PdfReader

MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PAGES = 60
MAX_CHARS = 200_000


class DocumentTooLarge(ValueError):
    """Input exceeded a size or page cap. Distinct from an unreadable file."""


def text_from_pdf_bytes(data: bytes) -> str:
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise DocumentTooLarge(
            f"PDF is {len(data) // (1024 * 1024)} MB; the limit is "
            f"{MAX_PDF_BYTES // (1024 * 1024)} MB."
        )

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            # Try the empty password, which unlocks most "protected" leases.
            # Anything still locked needs a password we do not have.
            try:
                if reader.decrypt("") == 0:
                    raise ValueError(
                        "This PDF is password-protected. Remove the password and "
                        "upload it again, or paste the text."
                    )
            except ValueError:
                raise
            except Exception as err:  # noqa: BLE001 - pypdf raises several types
                raise ValueError(f"Could not unlock this PDF: {err}") from err

        if len(reader.pages) > MAX_PAGES:
            raise DocumentTooLarge(
                f"PDF has {len(reader.pages)} pages; the limit is {MAX_PAGES}. "
                "Split the lease or paste the relevant pages."
            )

        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except (ValueError, DocumentTooLarge):
        raise
    except Exception as err:  # noqa: BLE001 - pypdf raises several parse errors
        raise ValueError(f"Could not read PDF: {err}") from err

    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No extractable text in this PDF. Scanned/image-only leases need OCR, "
            "which is out of scope for this project."
        )
    if len(text) > MAX_CHARS:
        raise DocumentTooLarge(
            f"Extracted text is {len(text):,} characters; the limit is "
            f"{MAX_CHARS:,}. Split the lease or paste the relevant pages."
        )
    return text
