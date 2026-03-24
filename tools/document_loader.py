"""Load PDF files and web URLs into LangChain ``Document`` objects for indexing."""

from __future__ import annotations

import logging
from typing import List, Literal

from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PdfEngine = Literal["pdfplumber", "pymupdf", "auto"]


def load_pdf(
    path: str,
    *,
    engine: PdfEngine = "auto",
    chunk: bool = True,
) -> List[Document]:
    """
    Load one PDF.

    ``auto`` (default): quick table count via pdfplumber; if the count reaches
    ``config.PDF_AUTO_MIN_TABLES`` (within ``PDF_AUTO_TABLE_SCAN_MAX_PAGES`` pages),
    uses ``tools/pdf_plumber_loader``; otherwise PyMuPDF (faster, weaker on tables).

    ``pdfplumber``: always table-aware extraction (Markdown tables + text chunks).

    ``pymupdf``: 페이지별 로드 후, ``chunk=True``이면 pdf_preprocessor와 동일한
    ``RecursiveCharacterTextSplitter``로 재분할 (페이지당 1청크가 아님).
    """
    resolved = engine
    if engine == "auto":
        from tools.pdf_plumber_loader import is_table_heavy_pdf

        resolved = "pdfplumber" if is_table_heavy_pdf(path) else "pymupdf"
        logger.info("PDF %s — auto engine: %s", path, resolved)

    if resolved == "pdfplumber":
        from tools.pdf_plumber_loader import load_pdf_file

        docs = load_pdf_file(path, chunk=chunk)
        logger.info("Loaded PDF %s (%s chunks, pdfplumber)", path, len(docs))
        return docs

    try:
        loader = PyMuPDFLoader(path, mode="page")
        docs = loader.load()
    except ImportError as exc:
        # Keep ingestion resilient: if PyMuPDF is unavailable, use pdfplumber path.
        logger.warning("PyMuPDF unavailable for %s (%s) -> fallback to pdfplumber", path, exc)
        from tools.pdf_plumber_loader import load_pdf_file

        docs = load_pdf_file(path, chunk=chunk)
        logger.info("Loaded PDF %s (%s chunks, pdfplumber fallback)", path, len(docs))
        return docs
    if chunk:
        from tools.pdf_plumber_loader import chunk_pdf_documents

        docs = chunk_pdf_documents(docs)
    logger.info("Loaded PDF %s (%s chunks, pymupdf)", path, len(docs))
    return docs


def load_url(url: str) -> List[Document]:
    """Fetch and parse a web URL into documents."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    logger.info("Loaded URL %s (%s chunks)", url, len(docs))
    return docs


def load_pdf_paths(
    paths: list[str],
    *,
    engine: PdfEngine = "auto",
    chunk: bool = True,
) -> List[Document]:
    """Load multiple PDFs and concatenate."""
    out: List[Document] = []
    for p in paths:
        out.extend(load_pdf(p, engine=engine, chunk=chunk))
    return out


def load_urls(urls: list[str]) -> List[Document]:
    """Load multiple URLs and concatenate."""
    out: List[Document] = []
    for u in urls:
        out.extend(load_url(u))
    return out
