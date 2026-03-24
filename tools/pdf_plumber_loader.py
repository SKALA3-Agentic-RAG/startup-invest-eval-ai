"""
PDF load via pdfplumber: table extraction to Markdown + text chunks for FAISS.

청킹 전략 (참고 ``pdf_preprocessor.chunk_documents`` 와 동일):
    - ``metadata["type"] == "table"``: 분할하지 않음 (행·열 관계 보존).
      ``PDF_CHUNK_SIZE``의 3배를 넘으면 경고 로그.
    - 그 외(일반 텍스트, PyMuPDF 페이지 문서 등): ``RecursiveCharacterTextSplitter``
    - 분할 우선순위: 단락(\\n\\n) → 줄바꿈(\\n) → ``。`` · ``.`` · 느낌표/물음표 → 공백 → 문자 단위.
    - ``length_function=len`` (문자 수; 토큰 기준보다 언어 중립적).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)

TABLE_COL_SEP = " | "
TABLE_ROW_SEP = "\n"


def count_nonempty_tables(
    pdf_path: str | Path,
    *,
    max_pages: int | None = None,
) -> int:
    """
    Count non-empty tables (pdfplumber ``extract_tables``) for routing ``engine="auto"``.

    ``max_pages`` limits how many pages are scanned (from the start); ``None`` scans all.
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    total = 0
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages
        if max_pages is not None:
            pages = pages[:max_pages]
        for page in pages:
            for table in page.extract_tables() or []:
                if table:
                    total += 1
    return total


def is_table_heavy_pdf(pdf_path: str | Path) -> bool:
    """True if this PDF should use the full pdfplumber pipeline (tables → Markdown, etc.)."""
    n = count_nonempty_tables(
        pdf_path,
        max_pages=config.PDF_AUTO_TABLE_SCAN_MAX_PAGES,
    )
    return n >= config.PDF_AUTO_MIN_TABLES


def _table_to_markdown(table: list[list[Optional[str]]]) -> str:
    """Turn pdfplumber ``extract_tables()`` rows into a GitHub-flavored Markdown table."""
    rows: list[str] = []
    for i, row in enumerate(table):
        cleaned = [
            re.sub(r"\s+", " ", cell.strip()) if cell else "" for cell in row
        ]
        rows.append(TABLE_COL_SEP.join(cleaned))
        if i == 0:
            rows.append(TABLE_COL_SEP.join(["---"] * len(row)))
    return TABLE_ROW_SEP.join(rows)


def _clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _strip_overlapping_table_cells(raw_text: str, tables: list[list[list[Optional[str]]]]) -> str:
    """Remove cell strings from page text once each to reduce table/body duplication."""
    for table in tables:
        if not table:
            continue
        for row in table:
            for cell in row:
                s = (cell or "").strip()
                if len(s) >= 2:
                    raw_text = raw_text.replace(s, "", 1)
    return _clean_text(raw_text)


def load_pdf_as_documents(pdf_path: str | Path) -> List[Document]:
    """
    Parse one PDF with pdfplumber: tables as separate ``Document`` rows (Markdown),
    remaining page text as another document when non-empty.
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Loading PDF with pdfplumber: %s", path.name)
    documents: List[Document] = []
    src = str(path)

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []

            for table in tables:
                if not table:
                    continue
                md_table = _table_to_markdown(table)
                if md_table.strip():
                    documents.append(
                        Document(
                            page_content=md_table,
                            metadata={
                                "source": src,
                                "page": page_num,
                                "type": "table",
                            },
                        )
                    )

            raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            if tables:
                cleaned = _strip_overlapping_table_cells(raw_text, tables)
            else:
                cleaned = _clean_text(raw_text)

            if cleaned:
                documents.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            "source": src,
                            "page": page_num,
                            "type": "text",
                        },
                    )
                )

    logger.info("pdfplumber extracted %s row(s) (pre-chunk)", len(documents))
    return documents


def _pdf_recursive_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Reference ``pdf_preprocessor.chunk_documents`` 와 동일한 분할기 설정.

    ``keep_separator=False``: 구분 문자를 청크 끝에 붙이지 않아 임베딩용 텍스트를 단순화.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.PDF_CHUNK_SIZE,
        chunk_overlap=config.PDF_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "！", "？", "!", "?", " ", ""],
        length_function=len,
        is_separator_regex=False,
        keep_separator=False,
    )


def chunk_pdf_documents(documents: List[Document]) -> List[Document]:
    """
    ``pdf_preprocessor.chunk_documents`` 와 동일한 규칙으로 청킹.

    - ``type == "table"``: 분할 없음, ``chunk_index=0``.
    - 그 외: ``RecursiveCharacterTextSplitter.split_documents``.
    """
    chunk_size = config.PDF_CHUNK_SIZE
    splitter = _pdf_recursive_text_splitter()

    out: List[Document] = []
    for doc in documents:
        if doc.metadata.get("type") == "table":
            if len(doc.page_content) > chunk_size * 3:
                logger.warning(
                    "[%s p%s] 표 크기(%s자)가 매우 큽니다. 필요 시 수동 분할을 권장합니다.",
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    len(doc.page_content),
                )
            doc.metadata["chunk_index"] = 0
            out.append(doc)
            continue

        splits = splitter.split_documents([doc])
        for idx, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = idx
        out.extend(splits)

    logger.info("청킹 완료: %s개 → %s개 청크", len(documents), len(out))
    return out


def chunk_documents(documents: List[Document]) -> List[Document]:
    """참고 모듈과 동일한 이름의 별칭."""
    return chunk_pdf_documents(documents)


def load_pdf_file(pdf_path: str | Path, *, chunk: bool = True) -> List[Document]:
    """Load one PDF with pdfplumber; optionally apply table-aware chunking."""
    raw = load_pdf_as_documents(pdf_path)
    return chunk_pdf_documents(raw) if chunk else raw
