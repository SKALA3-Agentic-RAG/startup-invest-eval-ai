"""LangGraph shared state: ``TypedDict`` + ``Annotated`` (docs + list reducers)."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from typing_extensions import Doc, TypeAlias

# --- Checkpoint-serialized shapes (``model_dump()`` JSON; matches schemas/*) ---

StartupProfileDict: TypeAlias = dict[str, Any]
"""JSON-compatible row matching :class:`schemas.startup.StartupProfile`."""

TechEvalDict: TypeAlias = dict[str, Any]
"""JSON-compatible row matching :class:`schemas.evaluation.TechEval`."""

MarketEvalDict: TypeAlias = dict[str, Any]
"""JSON-compatible row matching :class:`schemas.evaluation.MarketEval`."""

ScoreCardDict: TypeAlias = dict[str, Any]
"""JSON-compatible row matching :class:`schemas.evaluation.ScoreCard`."""

InvestDecisionDict: TypeAlias = dict[str, Any]
"""JSON-compatible row matching :class:`schemas.report.InvestDecision`."""


class GraphState(TypedDict, total=False):
    """
    Merged workflow state. Pydantic models are stored as dicts for SQLite checkpoints.

    Fields that accumulate across nodes use ``Annotated[..., operator.add]`` (list concat).
    Human-readable semantics for IDEs and type checkers use :class:`typing_extensions.Doc`.
    """

    query: Annotated[
        str,
        Doc("User research topic from the CLI ``--query`` argument."),
    ]
    report_date: Annotated[
        str,
        Doc("Report date string (YYYY-MM-DD) injected by ``main`` and used by ``report_agent``."),
    ]
    macro_context: Annotated[
        str | None,
        Doc("Macro and AI-sector context produced by ``macro_agent``."),
    ]
    startups: Annotated[
        list[StartupProfileDict],
        Doc(
            "Validated startup candidates after FAISS retrieval and web enrichment; "
            "each item is a ``StartupProfile`` dict."
        ),
    ]
    current_index: Annotated[
        int,
        Doc(
            "Legacy pointer; ``search_agent`` resets to 0. "
            "Tech/market evaluation now runs in parallel batches (no index loop)."
        ),
    ]
    tech_evals: Annotated[
        list[TechEvalDict],
        Doc(
            "Technical analysis contexts (one dict per startup). Filled by "
            "``parallel_startup_eval`` together with ``market_evals``; reducer: "
            "``operator.add``."
        ),
        operator.add,
    ]
    market_evals: Annotated[
        list[MarketEvalDict],
        Doc(
            "Market analysis contexts (one dict per startup). Filled in parallel with "
            "``tech_evals``; ``decision_agent`` consumes both sets. Reducer: "
            "``operator.add``."
        ),
        operator.add,
    ]
    scores: Annotated[
        list[ScoreCardDict],
        Doc("Weighted scores and ranks produced in ``decision_agent``."),
    ]
    invest_decisions: Annotated[
        list[InvestDecisionDict],
        Doc("GO / WATCH / PASS decisions from ``decision_agent``."),
    ]
    final_report: Annotated[
        str | None,
        Doc("Investment memo Markdown from ``report_agent``; written to disk by ``main``."),
    ]
    error: Annotated[
        str | None,
        Doc("Optional error summary when a node catches and records a failure."),
    ]
