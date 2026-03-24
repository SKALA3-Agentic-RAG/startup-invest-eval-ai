"""Application configuration: load secrets from `.env` and expose constants."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
# Optional; required only for private/gated Hub models
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

LLM_MODEL: str = "gpt-4o"
# Hugging Face Sentence Transformers — must match ingestion and search
EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-Embedding-0.6B"
QWEN_EMBEDDING_DEVICE: str | None = os.getenv("QWEN_EMBEDDING_DEVICE")  # e.g. "cuda", "cpu", "mps"
FAISS_INDEX_PATH: Path = _PROJECT_ROOT / "output" / "vectordb"
FAISS_INDEX_NAME: str = "startup_index"
REPORT_OUTPUT_PATH: Path = _PROJECT_ROOT / "output" / "reports"
CHECKPOINT_DB_PATH: Path = _PROJECT_ROOT / "output" / "checkpoints.db"
SCORE_WEIGHT_TECH: float = 0.4
SCORE_WEIGHT_MARKET: float = 0.6
MAX_STARTUPS: int = 10


@lru_cache
def get_chat_llm():
    """Return a temperature-0 chat model for structured extraction."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)


@lru_cache
def get_embeddings():
    """Hugging Face Qwen3 embeddings used for FAISS (see ``tools/qwen_embeddings.py``)."""
    from tools.qwen_embeddings import Qwen3HuggingFaceEmbeddings

    return Qwen3HuggingFaceEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        device=QWEN_EMBEDDING_DEVICE or None,
        token=HF_TOKEN,
    )
