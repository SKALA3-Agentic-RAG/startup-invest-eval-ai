"""Generate the final structured Markdown report."""

from __future__ import annotations

import json
import logging

import config
from agents.state import GraphState
from prompts.report import PROMPT
from schemas.report import FinalReport

logger = logging.getLogger(__name__)


def report_agent(state: GraphState) -> dict:
    """Ask the LLM for a ``FinalReport`` and store ``full_report_md`` in state."""
    logger.info("enter report_agent")
    try:
        llm = config.get_chat_llm().with_structured_output(FinalReport)
        chain = PROMPT | llm
        fr: FinalReport = chain.invoke(
            {
                "query": state.get("query", ""),
                "decisions_json": json.dumps(state.get("invest_decisions", []), ensure_ascii=False),
                "scores_json": json.dumps(state.get("scores", []), ensure_ascii=False),
                "startups_json": json.dumps(state.get("startups", []), ensure_ascii=False),
            }
        )
        logger.info("exit report_agent")
        return {"final_report": fr.full_report_md}
    except Exception as exc:  # noqa: BLE001
        logger.exception("report_agent failed")
        return {"final_report": None, "error": f"report_agent: {exc!s}"}
