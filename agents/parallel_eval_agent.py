"""Run tech and market evaluations in parallel (independent; joined by ``company_name`` in scoring)."""

from __future__ import annotations

import asyncio
import logging

from agents.market_agent import _run_market_eval_for_startup
from agents.state import GraphState, MarketEvalDict, TechEvalDict
from agents.tech_agent import _run_tech_eval_for_startup

logger = logging.getLogger(__name__)


async def parallel_startup_eval(state: GraphState) -> dict:
    """
    For every startup, run technical and market evaluations **at the same time**.

    All per-startup tasks use :func:`asyncio.to_thread` (sync LLM/RAG). Tech and market
    batches are awaited together with :func:`asyncio.gather`, so there is no ordering
    constraint between the two dimensions—``scoring_agent`` matches rows by
    ``company_name`` only.

    TODO: Cap concurrency with ``asyncio.Semaphore`` if APIs rate-limit.
    """
    logger.info("enter parallel_startup_eval")
    try:
        startups = state.get("startups") or []
        if not startups:
            logger.info("exit parallel_startup_eval (no startups)")
            return {}

        tech_tasks = [
            asyncio.to_thread(_run_tech_eval_for_startup, state, raw) for raw in startups
        ]
        market_tasks = [
            asyncio.to_thread(_run_market_eval_for_startup, state, raw) for raw in startups
        ]

        tech_results, market_results = await asyncio.gather(
            asyncio.gather(*tech_tasks),
            asyncio.gather(*market_tasks),
        )
        tech_list: list[TechEvalDict] = list(tech_results)
        market_list: list[MarketEvalDict] = list(market_results)

        logger.info(
            "exit parallel_startup_eval (tech=%s market=%s)",
            len(tech_list),
            len(market_list),
        )
        return {"tech_evals": tech_list, "market_evals": market_list}
    except Exception as exc:  # noqa: BLE001
        logger.exception("parallel_startup_eval failed")
        return {"error": f"parallel_startup_eval: {exc!s}"}
