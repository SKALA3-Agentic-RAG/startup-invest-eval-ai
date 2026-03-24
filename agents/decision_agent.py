"""LLM-based scoring + GO / WATCH / PASS decisions from tech/market analysis contexts."""

from __future__ import annotations

import json
import logging

import config
from agents.state import GraphState
from prompts.decision import PROMPT
from schemas.report import InvestDecisionsBatch

logger = logging.getLogger(__name__)


def decision_agent(state: GraphState) -> dict:
    """Produce score cards + structured investment decisions from tech/market analyses."""
    logger.info("enter decision_agent")
    try:
        tech_rows = state.get("tech_evals", [])
        market_rows = state.get("market_evals", [])
        llm = config.get_chat_llm().with_structured_output(InvestDecisionsBatch)
        chain = PROMPT | llm
        batch: InvestDecisionsBatch = chain.invoke(
            {
                "macro_context": state.get("macro_context") or "",
                "scores_json": "[]",
                "tech_json": json.dumps(tech_rows, ensure_ascii=False),
                "market_json": json.dumps(market_rows, ensure_ascii=False),
                "score_weight_tech": config.SCORE_WEIGHT_TECH,
                "score_weight_market": config.SCORE_WEIGHT_MARKET,
            }
        )
        ranked = sorted(batch.decisions, key=lambda d: d.total_score, reverse=True)
        ranked_out = []
        score_cards = []
        for i, d in enumerate(ranked, start=1):
            ranked_out.append(
                d.model_copy(update={"rank": i}).model_dump()
            )
            score_cards.append(
                {
                    "company_name": d.company_name,
                    "tech_score": d.tech_score,
                    "market_score": d.market_score,
                    "total_score": d.total_score,
                    "rank": i,
                }
            )
        logger.info("exit decision_agent (%s decisions)", len(batch.decisions))
        return {"invest_decisions": ranked_out, "scores": score_cards}
    except Exception as exc:  # noqa: BLE001
        logger.exception("decision_agent failed")
        return {"invest_decisions": [], "error": f"decision_agent: {exc!s}"}
