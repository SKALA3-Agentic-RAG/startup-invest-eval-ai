"""LangGraph shared state definition."""

import operator
from typing import Annotated, Any, TypedDict


class GraphState(TypedDict, total=False):
    """State passed between workflow nodes (lists store serialized Pydantic dicts)."""

    query: str
    macro_context: str | None
    startups: list[dict[str, Any]]
    current_index: int
    tech_evals: Annotated[list[dict[str, Any]], operator.add]
    market_evals: Annotated[list[dict[str, Any]], operator.add]
    scores: list[dict[str, Any]]
    invest_decisions: list[dict[str, Any]]
    final_report: str | None
    error: str | None
