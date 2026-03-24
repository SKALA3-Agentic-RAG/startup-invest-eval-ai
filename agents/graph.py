"""LangGraph ``StateGraph`` wiring for the agentic RAG investment workflow."""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.decision_agent import decision_agent
from agents.macro_agent import macro_agent
from agents.parallel_eval_agent import parallel_startup_eval
from agents.report_agent import report_agent
from agents.scoring_agent import scoring_agent
from agents.search_agent import search_agent
from agents.state import GraphState

logger = logging.getLogger(__name__)


def route_after_search(state: GraphState) -> Literal["startup_eval", "scoring"]:
    """Skip evaluation when no candidates remain."""
    startups = state.get("startups") or []
    if not startups:
        logger.info("route_after_search -> scoring (no startups)")
        return "scoring"
    logger.info("route_after_search -> startup_eval (tech + market in parallel)")
    return "startup_eval"


def build_graph(checkpointer):
    """
    Compile the workflow with the given checkpointer (e.g. ``AsyncSqliteSaver``).

    Flow: macro → search → startup_eval (tech ∥ market per startup) → scoring →
    decision → report → END.
    """
    builder = StateGraph(GraphState)
    builder.add_node("macro", macro_agent)
    builder.add_node("search", search_agent)
    builder.add_node("startup_eval", parallel_startup_eval)
    builder.add_node("scoring", scoring_agent)
    builder.add_node("decision", decision_agent)
    builder.add_node("report", report_agent)

    builder.set_entry_point("macro")
    builder.add_edge("macro", "search")
    builder.add_conditional_edges(
        "search",
        route_after_search,
        {"startup_eval": "startup_eval", "scoring": "scoring"},
    )
    builder.add_edge("startup_eval", "scoring")
    builder.add_edge("scoring", "decision")
    builder.add_edge("decision", "report")
    builder.add_edge("report", END)

    return builder.compile(checkpointer=checkpointer)
