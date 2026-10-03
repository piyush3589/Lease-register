"""Write text-layer sample PDFs (no extra PDF library — Helvetica Type1)."""

from pathlib import Path


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_line(line: str, width: int = 88) -> list[str]:
    if not line:
        return [""]
    words = line.split(" ")
    rows: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if len(trial) <= width:
            current = trial
            continue
        if current:
            rows.append(current)
        current = word
    if current:
        rows.append(current)
    return rows


def page_content_stream(lines: list[str], start_y: float = 740.0) -> str:
    parts = ["BT", "/F1 11 Tf", "16 TL", f"72 {start_y} Td"]
    first = True
    for line in lines:
        escaped = pdf_escape(line) if line else ""
        if first:
            parts.append(f"({escaped}) Tj")
            first = False
        else:
            parts.append("T*")
            parts.append(f"({escaped}) Tj")
    parts.append("ET")
    return "\n".join(parts) + "\n"


def build_pdf(text: str) -> bytes:
    raw_lines: list[str] = []
    for paragraph in text.replace("\r\n", "\n").split("\n"):
        raw_lines.extend(wrap_line(paragraph.rstrip()))

    stream = page_content_stream(raw_lines)
    stream_bytes = stream.encode("latin-1", errors="replace")
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream_bytes)} >>\nstream\n{stream}endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{index} 0 obj\n{body}\nendobj\n".encode("latin-1"))

    xref_at = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(out)


def main() -> None:
    folder = Path(__file__).resolve().parent / "sample_leases"
    for txt in sorted(folder.glob("*.txt")):
        pdf_path = txt.with_suffix(".pdf")
        pdf_path.write_bytes(build_pdf(txt.read_text(encoding="utf-8")))
        print(f"wrote {pdf_path.name}")


if __name__ == "__main__":
    main()
