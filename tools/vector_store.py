"""FAISS vector store helpers (LangChain community + Hugging Face Qwen3 embeddings)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


def _store_path(path: Optional[str] = None) -> str:
    return path if path is not None else str(config.FAISS_INDEX_PATH)


def _try_load_index(path: Optional[str] = None) -> Optional[FAISS]:
    p = _store_path(path)
    try:
        return load_index(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FAISS load skipped (%s): %s", p, exc)
        return None


def build_index(docs: List[Document]) -> FAISS:
    """Embed documents and build an in-memory FAISS index."""
    embeddings = config.get_embeddings()
    return FAISS.from_documents(docs, embeddings)


def save_index(faiss_store: FAISS, path: str) -> None:
    """Persist FAISS index under ``path`` using ``config.FAISS_INDEX_NAME``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    faiss_store.save_local(folder_path=str(p), index_name=config.FAISS_INDEX_NAME)
    logger.info("Saved FAISS index to %s (%s)", p, config.FAISS_INDEX_NAME)


def load_index(path: str) -> FAISS:
    """Load a FAISS index from disk; raises if missing."""
    embeddings = config.get_embeddings()
    return FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        index_name=config.FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )


def search(query: str, k: int, *, path: Optional[str] = None) -> List[Document]:
    """Similarity search against the on-disk index."""
    store = _try_load_index(path)
    if store is None:
        return []
    return store.similarity_search(query, k=k)


def max_marginal_relevance_search(
    query: str,
    k: int,
    fetch_k: int = 20,
    *,
    path: Optional[str] = None,
) -> List[Document]:
    """MMR search over the persisted index (diversity vs pure similarity)."""
    store = _try_load_index(path)
    if store is None:
        return []
    return store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def similarity_search_with_score(
    query: str,
    k: int,
    *,
    path: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """Top-``k`` documents with similarity scores (inner product with normalized embeddings ≈ cosine)."""
    store = _try_load_index(path)
    if store is None:
        return []
    return store.similarity_search_with_score(query, k=k)


def filtered_search(
    query: str,
    k: int,
    *,
    doc_type: Optional[str] = None,
    source: Optional[str] = None,
    path: Optional[str] = None,
    candidate_multiplier: int = 10,
) -> List[Document]:
    """
    Post-filter similarity search (FAISS has no native metadata filters).

    ``source`` matches the document's ``metadata["source"]`` basename or full path string.
    ``doc_type`` matches ``metadata["type"]`` (e.g. ``\"table\"``, ``\"text\"``) from pdfplumber ingestion.
    """
    store = _try_load_index(path)
    if store is None:
        return []

    take = max(k * candidate_multiplier, k)
    candidates = store.similarity_search(query, k=take)

    def source_ok(meta_src: str) -> bool:
        if not source:
            return True
        if meta_src == source:
            return True
        return Path(meta_src).name == source

    out: List[Document] = []
    for doc in candidates:
        if doc_type and doc.metadata.get("type") != doc_type:
            continue
        if source and not source_ok(str(doc.metadata.get("source", ""))):
            continue
        out.append(doc)
        if len(out) >= k:
            break
    return out


def get_index_stats(path: Optional[str] = None) -> Optional[dict]:
    """Return FAISS index stats, or ``None`` if the index cannot be loaded."""
    store = _try_load_index(path)
    if store is None:
        return None
    idx = store.index
    return {
        "total_vectors": int(idx.ntotal),
        "index_type": type(idx).__name__,
        "embedding_dim": int(idx.d),
        "index_dir": _store_path(path),
        "index_name": config.FAISS_INDEX_NAME,
    }
