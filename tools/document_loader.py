"""Load PDF files and web URLs into LangChain ``Document`` objects for indexing."""

from __future__ import annotations

import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(path: str) -> List[Document]:
    """Load a single PDF path into documents (one chunk per page by default)."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    logger.info("Loaded PDF %s (%s chunks)", path, len(docs))
    return docs


def load_url(url: str) -> List[Document]:
    """Fetch and parse a web URL into documents."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    logger.info("Loaded URL %s (%s chunks)", url, len(docs))
    return docs


def load_pdf_paths(paths: list[str]) -> List[Document]:
    """Load multiple PDFs and concatenate."""
    out: List[Document] = []
    for p in paths:
        out.extend(load_pdf(p))
    return out


def load_urls(urls: list[str]) -> List[Document]:
    """Load multiple URLs and concatenate."""
    out: List[Document] = []
    for u in urls:
        out.extend(load_url(u))
    return out
