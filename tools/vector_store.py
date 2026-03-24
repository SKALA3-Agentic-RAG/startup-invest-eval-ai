"""FAISS vector store helpers (LangChain community + Hugging Face Qwen3 embeddings)."""

from __future__ import annotations

import logging
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


def build_index(docs: List[Document]) -> FAISS:
    """Embed documents and build an in-memory FAISS index."""
    embeddings = config.get_embeddings()
    return FAISS.from_documents(docs, embeddings)


def save_index(faiss_store: FAISS, path: str) -> None:
    """Persist FAISS index under ``path`` using ``config.FAISS_INDEX_NAME``."""
    from pathlib import Path

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


def search(query: str, k: int) -> List[Document]:
    """Similarity search against the on-disk startup index."""
    path = str(config.FAISS_INDEX_PATH)
    try:
        store = load_index(path)
    except Exception as exc:  # noqa: BLE001 — surface as empty results + log
        logger.warning("FAISS load/search skipped (%s): %s", path, exc)
        return []
    return store.similarity_search(query, k=k)
