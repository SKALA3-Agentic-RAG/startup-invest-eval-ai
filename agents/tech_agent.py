"""Technical deep-dive using hybrid RAG context."""

from __future__ import annotations

import json
import logging

import config
from agents.state import GraphState
from prompts.tech_eval import PROMPT
from schemas.evaluation import TechEval
from schemas.startup import StartupProfile
from tools import retriever

logger = logging.getLogger(__name__)


def _format_docs(docs: list) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        src = (d.metadata or {}).get("source", "")
        parts.append(f"[{i}] ({src})\n{d.page_content}")
    return "\n\n".join(parts)[:24000]


def tech_agent(state: GraphState) -> dict:
    """Run technical evaluation for ``startups[current_index]``."""
    logger.info("enter tech_agent")
    try:
        startups = state.get("startups") or []
        idx = int(state.get("current_index", 0))
        if idx >= len(startups):
            logger.info("exit tech_agent (no startup at index)")
            return {}
        raw = startups[idx]
        profile = StartupProfile.model_validate(raw)
        ctx_docs = retriever.merge_context(
            query=state.get("query", ""),
            company_name=profile.company_name,
        )
        context = _format_docs(ctx_docs)
        llm = config.get_chat_llm().with_structured_output(TechEval)
        chain = PROMPT | llm
        te: TechEval = chain.invoke(
            {
                "macro_context": state.get("macro_context") or "",
                "startup_json": json.dumps(raw, ensure_ascii=False),
                "context": context,
            }
        )
        logger.info("exit tech_agent")
        return {"tech_evals": [te.model_dump()]}
    except Exception as exc:  # noqa: BLE001
        logger.exception("tech_agent failed")
        return {"error": f"tech_agent: {exc!s}"}
