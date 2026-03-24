"""Hugging Face ``Qwen/Qwen3-Embedding-0.6B`` via ``sentence-transformers``."""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _as_2d_list(arr: np.ndarray) -> List[List[float]]:
    """Convert a 2-D float array to plain Python lists for FAISS / LangChain."""
    return np.asarray(arr, dtype=np.float32).tolist()


class Qwen3HuggingFaceEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings for ``Qwen/Qwen3-Embedding-0.6B``.

    Follows the model card: queries use ``prompt_name="query"``; corpus / documents
    are encoded without that prompt for asymmetric retrieval.

    Requires ``transformers>=4.51.0`` and ``sentence-transformers>=2.7.0``.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        token: str | None = None,
    ) -> None:
        """
        Load the SentenceTransformer wrapper for the given Hub model id.

        Args:
            model_id: Hugging Face repo id (e.g. ``Qwen/Qwen3-Embedding-0.6B``).
            device: Optional torch device string; ``None`` lets the library decide.
            token: Hub token for private/gated models; falls back to env vars.
        """
        self.model_id = model_id
        tok = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        kwargs: dict = {}
        if device:
            kwargs["device"] = device
        # TODO: Optional speedups — ``model_kwargs={"attn_implementation": "flash_attention_2"}`` on CUDA.
        self._model = SentenceTransformer(model_id, token=tok, **kwargs)
        logger.info("Loaded embedding model %s", model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Encode document/corpus strings (no query prompt)."""
        if not texts:
            return []
        vectors = self._model.encode(texts)
        return _as_2d_list(np.atleast_2d(vectors))

    def embed_query(self, text: str) -> List[float]:
        """Encode a single search query with the recommended ``query`` prompt."""
        vectors = self._model.encode([text], prompt_name="query")
        row = np.atleast_2d(vectors)[0]
        return row.astype(np.float32).tolist()
