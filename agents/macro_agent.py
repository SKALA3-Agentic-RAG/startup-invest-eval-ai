"""Macro environment analysis node."""

from __future__ import annotations

import logging

import config
from agents.state import GraphState
from prompts.macro import PROMPT
from schemas.evaluation import MacroAnalysis

logger = logging.getLogger(__name__)


def macro_agent(state: GraphState) -> dict:
    """Populate ``macro_context`` from the user query."""
    logger.info("enter macro_agent")
    try:
        llm = config.get_chat_llm().with_structured_output(MacroAnalysis)
        chain = PROMPT | llm
        out: MacroAnalysis = chain.invoke({"query": state.get("query", "")})
        logger.info("exit macro_agent")
        return {"macro_context": out.macro_context}
    except Exception as exc:  # noqa: BLE001
        logger.exception("macro_agent failed")
        return {"error": f"macro_agent: {exc!s}"}
