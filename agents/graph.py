"""LangGraph ``StateGraph`` wiring for the agentic RAG investment workflow."""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.decision_agent import decision_agent
from agents.macro_agent import macro_agent
from agents.market_agent import market_agent
from agents.report_agent import report_agent
from agents.scoring_agent import scoring_agent
from agents.search_agent import search_agent
from agents.state import GraphState
from agents.tech_agent import tech_agent
from memory.checkpointer import get_checkpointer

logger = logging.getLogger(__name__)


def route_after_search(state: GraphState) -> Literal["tech_eval", "scoring"]:
    """Skip per-startup loop when no candidates remain."""
    startups = state.get("startups") or []
    if not startups:
        logger.info("route_after_search -> scoring (no startups)")
        return "scoring"
    logger.info("route_after_search -> tech_eval")
    return "tech_eval"


def route_after_market(state: GraphState) -> Literal["tech_eval", "scoring"]:
    """Loop tech→market until ``current_index`` reaches ``len(startups)``."""
    startups = state.get("startups") or []
    idx = int(state.get("current_index", 0))
    if idx < len(startups):
        logger.info("route_after_market -> tech_eval (index %s / %s)", idx, len(startups))
        return "tech_eval"
    logger.info("route_after_market -> scoring")
    return "scoring"


def build_graph():
    """
    Compile the workflow with a SQLite checkpointer.

    Flow: macro → search → (tech_eval ↔ market_eval)* → scoring → decision → report → END.
    """
    builder = StateGraph(GraphState)
    builder.add_node("macro", macro_agent)
    builder.add_node("search", search_agent)
    builder.add_node("tech_eval", tech_agent)
    builder.add_node("market_eval", market_agent)
    builder.add_node("scoring", scoring_agent)
    builder.add_node("decision", decision_agent)
    builder.add_node("report", report_agent)

    builder.set_entry_point("macro")
    builder.add_edge("macro", "search")
    builder.add_conditional_edges(
        "search",
        route_after_search,
        {"tech_eval": "tech_eval", "scoring": "scoring"},
    )
    builder.add_edge("tech_eval", "market_eval")
    builder.add_conditional_edges(
        "market_eval",
        route_after_market,
        {"tech_eval": "tech_eval", "scoring": "scoring"},
    )
    builder.add_edge("scoring", "decision")
    builder.add_edge("decision", "report")
    builder.add_edge("report", END)

    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)
