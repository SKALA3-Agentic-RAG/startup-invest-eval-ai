"""Hybrid retrieval: FAISS MMR plus Tavily results merged and URL-deduped."""

from __future__ import annotations

import logging
from typing import List
from urllib.parse import urlparse

from langchain_core.documents import Document

import config
import tools.vector_store as vector_store
from tools import web_search

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return f"{parsed.netloc}{parsed.path}".rstrip("/").lower()


def faiss_mmr_documents(query: str, k: int = 5, fetch_k: int = 20) -> List[Document]:
    """Maximal marginal relevance over the persisted FAISS index."""
    return vector_store.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, path=str(config.FAISS_INDEX_PATH)
    )


def web_results_as_documents(query: str) -> List[Document]:
    """Convert Tavily hits into ``Document`` rows with ``source`` metadata."""
    hits = web_search.search(query)
    docs: List[Document] = []
    for h in hits:
        url = h.get("url") or ""
        text = h.get("content") or h.get("snippet") or str(h)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": url, "title": h.get("title", "")},
            )
        )
    return docs


def merge_context(
    *,
    query: str,
    company_name: str,
    k: int = 5,
    fetch_k: int = 20,
) -> List[Document]:
    """
    Combine FAISS MMR with a company-focused web search, deduplicating by URL.

    TODO: Add re-ranking or cross-encoder filtering for higher precision.
    """
    faiss_q = f"{query} {company_name}".strip()
    web_q = f"{company_name} startup AI technology product latest news"
    mmr = faiss_mmr_documents(faiss_q, k=k, fetch_k=fetch_k)
    web_docs = web_results_as_documents(web_q)

    seen: set[str] = set()
    merged: List[Document] = []
    for doc in mmr + web_docs:
        key = _normalize_url(str(doc.metadata.get("source", "")))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(doc)
    return merged
