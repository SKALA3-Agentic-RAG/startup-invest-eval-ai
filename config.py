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
# PDFs to embed into FAISS (see ``tools/ingest_pdfs.py``)
_pdf_src_env = os.getenv("PDF_SOURCE_DIR")
PDF_SOURCE_DIR: Path = (
    Path(_pdf_src_env).expanduser().resolve()
    if _pdf_src_env
    else (_PROJECT_ROOT / "data" / "pdfs")
)
REPORT_OUTPUT_PATH: Path = _PROJECT_ROOT / "output" / "reports"
REPORT_PDF_OUTPUT_PATH: Path = _PROJECT_ROOT / "outputs" / "pdfs"
CHECKPOINT_DB_PATH: Path = _PROJECT_ROOT / "output" / "checkpoints.db"
# Sentence Transformers / Hugging Face Hub weights (not ~/.cache)
_hf_cache_env = os.getenv("HF_MODEL_CACHE_PATH")
HF_MODEL_CACHE_PATH: Path = (
    Path(_hf_cache_env).expanduser().resolve()
    if _hf_cache_env
    else (_PROJECT_ROOT / "models" / "huggingface")
)
# If true, embeddings are loaded strictly from local cache (no Hub network calls).
HF_LOCAL_FILES_ONLY: bool = os.getenv("HF_LOCAL_FILES_ONLY", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
SCORE_WEIGHT_TECH: float = 0.4
SCORE_WEIGHT_MARKET: float = 0.6
MAX_STARTUPS: int = 10
# OpenAI rate-limit control: cap concurrency + retry/backoff.
MAX_PARALLEL_STARTUP_EVALS: int = max(1, int(os.getenv("MAX_PARALLEL_STARTUP_EVALS", "2")))
MAX_PARALLEL_SEARCH_ENRICH: int = max(1, int(os.getenv("MAX_PARALLEL_SEARCH_ENRICH", "3")))
OPENAI_RETRY_MAX_ATTEMPTS: int = max(1, int(os.getenv("OPENAI_RETRY_MAX_ATTEMPTS", "4")))
OPENAI_RETRY_BASE_SECONDS: float = max(0.1, float(os.getenv("OPENAI_RETRY_BASE_SECONDS", "1.5")))
# PDF 청킹 (``tools/pdf_plumber_loader.chunk_pdf_documents`` — 참고 pdf_preprocessor와 동일)
# chunk_size: Qwen3 임베딩 최대 길이(토큰) 대비 문자 기준 512 — 검색 해상도 균형.
# chunk_overlap: 경계 문맥 유지용 64자.
PDF_CHUNK_SIZE: int = int(os.getenv("PDF_CHUNK_SIZE", "512"))
PDF_CHUNK_OVERLAP: int = int(os.getenv("PDF_CHUNK_OVERLAP", "64"))
# engine="auto" in ``document_loader``: use pdfplumber when enough tables are detected
PDF_AUTO_MIN_TABLES: int = max(1, int(os.getenv("PDF_AUTO_MIN_TABLES", "2")))
_scan = os.getenv("PDF_AUTO_TABLE_SCAN_MAX_PAGES", "").strip()
if not _scan:
    PDF_AUTO_TABLE_SCAN_MAX_PAGES: int | None = None
else:
    _scan_n = int(_scan)
    PDF_AUTO_TABLE_SCAN_MAX_PAGES = None if _scan_n <= 0 else _scan_n


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
        cache_folder=str(HF_MODEL_CACHE_PATH),
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
