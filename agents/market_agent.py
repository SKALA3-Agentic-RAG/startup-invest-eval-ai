"""Market evaluation helpers (used by ``parallel_eval_agent``)."""

from __future__ import annotations

import json
import logging
from typing import Any

import config
from agents.state import GraphState, MarketEvalDict
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


def _run_market_eval_for_startup(state: GraphState, raw: dict[str, Any]) -> MarketEvalDict:
    """Synchronous market evaluation for one startup dict (runs in a worker thread)."""
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
    # Upstream scoring is moved to ``decision_agent``; keep market stage qualitative.
    analysis_context = (
        f"tam_usd_bn={me.tam_usd_bn}\n"
        f"growth_rate_pct={me.growth_rate_pct}\n"
        f"competition_level={me.competition_level}\n"
        f"rationale={me.rationale}"
    )
    return {
        "company_name": me.company_name,
        "analysis_context": analysis_context,
    }
