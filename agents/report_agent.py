"""Generate the final structured Markdown report."""

from __future__ import annotations

import json
import logging

import config
from agents.state import GraphState
from prompts.report import CITATION_FORMAT_GUIDE_KR, PROMPT
from schemas.report import FinalReport

logger = logging.getLogger(__name__)


def report_agent(state: GraphState) -> dict:
    """Ask the LLM for a ``FinalReport`` and store ``full_report_md`` in state."""
    logger.info("enter report_agent")
    try:
        scores_json = json.dumps(state.get("scores", []), ensure_ascii=False)
        decisions_json = json.dumps(state.get("invest_decisions", []), ensure_ascii=False)
        # The report prompt expects an "English evaluation result" blob. We provide
        # the latest structured outputs from decision/analysis stages as a single payload.
        english_evaluation_result = json.dumps(
            {
                "decisions": state.get("invest_decisions", []),
                "scores": state.get("scores", []),
                "tech_evals": state.get("tech_evals", []),
                "market_evals": state.get("market_evals", []),
            },
            ensure_ascii=False,
        )
        llm = config.get_chat_llm().with_structured_output(FinalReport)
        chain = PROMPT | llm
        fr: FinalReport = chain.invoke(
            {
                "query": state.get("query", ""),
                "report_date": state.get("report_date", ""),
                "decisions_json": decisions_json,
                "scores_json": scores_json,
                "startups_json": json.dumps(state.get("startups", []), ensure_ascii=False),
                "english_evaluation_result": english_evaluation_result,
                "target_domain": "Robotics & AI",
                "citation_format_guide": CITATION_FORMAT_GUIDE_KR,
            }
        )
        logger.info("exit report_agent")
        return {"final_report": fr.full_report_md}
    except Exception as exc:  # noqa: BLE001
        logger.exception("report_agent failed")
        return {"final_report": None, "error": f"report_agent: {exc!s}"}
