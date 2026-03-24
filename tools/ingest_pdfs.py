#!/usr/bin/env python3
"""Embed every ``*.pdf`` under ``data/pdfs/`` into the FAISS index at ``output/vectordb/``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
from tools import document_loader, vector_store
from tools.document_loader import PdfEngine

logger = logging.getLogger(__name__)


def _pdf_paths(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.glob("*.pdf") if p.is_file())


def ingest_from_folder(
    folder: Path | None = None,
    *,
    engine: PdfEngine = "auto",
) -> int:
    """
    Load all PDFs from ``folder``, build FAISS, save to ``config.FAISS_INDEX_PATH``.

    ``engine`` is passed to ``document_loader.load_pdf_paths`` (default ``auto``).

    Returns 0 on success, 1 if there is nothing to index or extraction yields no chunks.
    """
    root = folder if folder is not None else config.PDF_SOURCE_DIR
    root.mkdir(parents=True, exist_ok=True)

    paths = _pdf_paths(root)
    if not paths:
        logger.warning("No PDF files in %s — add .pdf files and run again.", root)
        return 1

    logger.info("Indexing %s PDF(s) from %s (engine=%s)", len(paths), root, engine)
    docs = document_loader.load_pdf_paths(paths, engine=engine)
    if not docs:
        logger.error("PDFs produced no document chunks.")
        return 1

    store = vector_store.build_index(docs)
    config.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_index(store, str(config.FAISS_INDEX_PATH))
    logger.info("Wrote FAISS index to %s", config.FAISS_INDEX_PATH)
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build FAISS from all PDFs in the project PDF folder (default: data/pdfs).",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Override PDF directory (default: PDF_SOURCE_DIR / data/pdfs).",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "pdfplumber", "pymupdf"),
        default="auto",
        help="auto: pdfplumber only for table-heavy PDFs (see config PDF_AUTO_*); "
        "pdfplumber: always; pymupdf: always fast page-based text.",
    )
    args = parser.parse_args()
    code = ingest_from_folder(args.dir, engine=args.engine)
    sys.exit(code)


if __name__ == "__main__":
    main()
