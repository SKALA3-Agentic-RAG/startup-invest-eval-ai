"""Export Markdown reports to PDF files."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


def _normalize_md_for_pdf(markdown_text: str) -> list[str]:
    """Normalize Markdown to simple line-oriented text for PDF output."""
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out: list[str] = []
    for ln in lines:
        s = ln.rstrip()
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        if s.startswith("- "):
            s = "• " + s[2:]
        out.append(s)
    return out


def markdown_to_pdf(markdown_text: str, pdf_path: str | Path) -> Path:
    """Write a Markdown string as a readable text PDF."""
    out = Path(pdf_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Built-in CID font available in reportlab for Korean text.
    font_name = "HYSMyeongJo-Medium"
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    page_w, page_h = A4
    margin_x = 40
    margin_y = 40
    line_h = 15

    c = canvas.Canvas(str(out), pagesize=A4)
    c.setAuthor("agentic-rag")
    c.setTitle(out.stem)
    c.setFont(font_name, 11)

    y = page_h - margin_y
    max_chars = int((page_w - 2 * margin_x) / 6.0)

    def write_line(s: str) -> None:
        nonlocal y
        if y <= margin_y:
            c.showPage()
            c.setFont(font_name, 11)
            y = page_h - margin_y
        c.drawString(margin_x, y, s)
        y -= line_h

    for line in _normalize_md_for_pdf(markdown_text):
        if not line:
            write_line("")
            continue
        chunk = line
        while len(chunk) > max_chars:
            write_line(chunk[:max_chars])
            chunk = chunk[max_chars:]
        write_line(chunk)

    c.save()
    return out

