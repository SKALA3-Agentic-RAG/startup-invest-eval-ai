"""LLM-based GO / WATCH / PASS decisions from scores and qualitative evals."""

from __future__ import annotations

import json
import logging

import config
from agents.state import GraphState
from prompts.decision import PROMPT
from schemas.report import InvestDecisionsBatch

logger = logging.getLogger(__name__)


def decision_agent(state: GraphState) -> dict:
    """Produce structured investment decisions for all scored companies."""
    logger.info("enter decision_agent")
    try:
        llm = config.get_chat_llm().with_structured_output(InvestDecisionsBatch)
        chain = PROMPT | llm
        batch: InvestDecisionsBatch = chain.invoke(
            {
                "macro_context": state.get("macro_context") or "",
                "scores_json": json.dumps(state.get("scores", []), ensure_ascii=False),
                "tech_json": json.dumps(state.get("tech_evals", []), ensure_ascii=False),
                "market_json": json.dumps(state.get("market_evals", []), ensure_ascii=False),
            }
        )
        logger.info("exit decision_agent (%s decisions)", len(batch.decisions))
        return {"invest_decisions": [d.model_dump() for d in batch.decisions]}
    except Exception as exc:  # noqa: BLE001
        logger.exception("decision_agent failed")
        return {"invest_decisions": [], "error": f"decision_agent: {exc!s}"}
