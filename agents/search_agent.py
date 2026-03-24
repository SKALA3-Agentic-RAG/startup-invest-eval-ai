"""FAISS retrieval + parallel Tavily validation / enrichment."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.documents import Document

import config
from agents.state import GraphState
from prompts.startup_validation import PROMPT
from schemas.startup import StartupProfile
from tools import vector_store, web_search

logger = logging.getLogger(__name__)


def _doc_to_seed(doc: Document) -> tuple[str, str]:
    """Derive (company_name, vector_snippet) from a retrieved document."""
    name = (doc.metadata or {}).get("company_name") or ""
    if not name:
        first = (doc.page_content or "").strip().split("\n", 1)[0]
        name = (first[:120] or "Unknown Company").strip()
    return name, doc.page_content or ""


def _validate_one(company_name: str, vector_snippet: str, web_hits: list[dict[str, Any]]) -> StartupProfile:
    llm = config.get_chat_llm().with_structured_output(StartupProfile)
    chain = PROMPT | llm
    return chain.invoke(
        {
            "company_name": company_name,
            "vector_snippet": vector_snippet[:8000],
            "web_hits": json.dumps(web_hits, ensure_ascii=False)[:12000],
        }
    )


async def _enrich_candidate(doc: Document) -> StartupProfile:
    company, snippet = _doc_to_seed(doc)
    q = f"{company} startup funding round AI"
    web_hits = await asyncio.to_thread(web_search.search, q)
    return await asyncio.to_thread(_validate_one, company, snippet, web_hits)


async def _search_async(state: GraphState) -> dict:
    query = state.get("query", "")
    docs = vector_store.search(query, k=10)
    if not docs:
        logger.warning("No FAISS hits — build an index under output/vectordb (see vector_store).")
        return {"startups": [], "current_index": 0}

    enriched = await asyncio.gather(*[_enrich_candidate(d) for d in docs])
    filtered = [p for p in enriched if p.is_startup][: config.MAX_STARTUPS]
    return {
        "startups": [p.model_dump() for p in filtered],
        "current_index": 0,
    }


def search_agent(state: GraphState) -> dict:
    """
    Retrieve top-k from FAISS, validate each candidate with Tavily + structured LLM,
    then keep only ``is_startup`` profiles.
    """
    logger.info("enter search_agent")
    try:
        out = asyncio.run(_search_async(state))
        logger.info("exit search_agent (kept %s startups)", len(out.get("startups", [])))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_agent failed")
        return {"startups": [], "current_index": 0, "error": f"search_agent: {exc!s}"}
