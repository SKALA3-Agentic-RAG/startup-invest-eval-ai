"""Deterministic weighted scoring and ranking across evaluated startups."""

from __future__ import annotations

import logging

import config
from agents.state import GraphState
from schemas.evaluation import ScoreCard

logger = logging.getLogger(__name__)


def scoring_agent(state: GraphState) -> dict:
    """
    Join tech and market evals by ``company_name``, compute weighted totals, assign ranks.

    TODO: Add sector-specific weights or calibration layers.
    """
    logger.info("enter scoring_agent")
    try:
        tech_by = {t["company_name"]: t for t in state.get("tech_evals", [])}
        mkt_by = {m["company_name"]: m for m in state.get("market_evals", [])}
        names = sorted(set(tech_by) & set(mkt_by))
        cards: list[ScoreCard] = []
        for name in names:
            t_score = float(tech_by[name]["tech_score"])
            m_score = float(mkt_by[name]["traction_score"])
            total = config.SCORE_WEIGHT_TECH * t_score + config.SCORE_WEIGHT_MARKET * m_score
            cards.append(
                ScoreCard(
                    company_name=name,
                    tech_score=t_score,
                    market_score=m_score,
                    total_score=total,
                    rank=0,
                )
            )
        cards.sort(key=lambda c: c.total_score, reverse=True)
        ranked: list[ScoreCard] = []
        for i, c in enumerate(cards, start=1):
            ranked.append(
                ScoreCard(
                    company_name=c.company_name,
                    tech_score=c.tech_score,
                    market_score=c.market_score,
                    total_score=c.total_score,
                    rank=i,
                )
            )
        logger.info("exit scoring_agent (%s score cards)", len(ranked))
        return {"scores": [c.model_dump() for c in ranked]}
    except Exception as exc:  # noqa: BLE001
        logger.exception("scoring_agent failed")
        return {"scores": [], "error": f"scoring_agent: {exc!s}"}
