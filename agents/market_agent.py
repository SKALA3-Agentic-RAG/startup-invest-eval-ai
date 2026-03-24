"""Market sizing and traction evaluation."""

from __future__ import annotations

import json
import logging

import config
from agents.state import GraphState
from prompts.market_eval import PROMPT
from schemas.evaluation import MarketEval
from schemas.startup import StartupProfile
from tools import retriever

logger = logging.getLogger(__name__)


def _format_docs(docs: list) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        src = (d.metadata or {}).get("source", "")
        parts.append(f"[{i}] ({src})\n{d.page_content}")
    return "\n\n".join(parts)[:24000]


def market_agent(state: GraphState) -> dict:
    """Evaluate market for the current startup and advance ``current_index``."""
    logger.info("enter market_agent")
    try:
        startups = state.get("startups") or []
        idx = int(state.get("current_index", 0))
        if idx >= len(startups):
            logger.info("exit market_agent (no startup at index)")
            return {}
        raw = startups[idx]
        profile = StartupProfile.model_validate(raw)
        ctx_docs = retriever.merge_context(
            query=state.get("query", ""),
            company_name=profile.company_name,
        )
        context = _format_docs(ctx_docs)
        llm = config.get_chat_llm().with_structured_output(MarketEval)
        chain = PROMPT | llm
        me: MarketEval = chain.invoke(
            {
                "macro_context": state.get("macro_context") or "",
                "startup_json": json.dumps(raw, ensure_ascii=False),
                "context": context,
            }
        )
        logger.info("exit market_agent (index %s -> %s)", idx, idx + 1)
        return {
            "market_evals": [me.model_dump()],
            "current_index": idx + 1,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("market_agent failed")
        return {"error": f"market_agent: {exc!s}"}
