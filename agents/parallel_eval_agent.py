"""Run tech and market analyses in parallel (joined later in ``decision_agent``)."""

from __future__ import annotations

import asyncio
import logging

import config
from agents.market_agent import _run_market_eval_for_startup
from agents.state import GraphState, MarketEvalDict, TechEvalDict
from agents.tech_agent import _run_tech_eval_for_startup

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: Exception) -> bool:
    txt = f"{type(exc).__name__}: {exc!s}".lower()
    return "rate limit" in txt or "ratelimit" in txt or "429" in txt


async def parallel_startup_eval(state: GraphState) -> dict:
    """
    For every startup, run technical and market evaluations **at the same time**.

    All per-startup tasks use :func:`asyncio.to_thread` (sync LLM/RAG). Tech and market
    batches are awaited together with :func:`asyncio.gather`, so there is no ordering
    constraint between the two dimensions—``decision_agent`` joins rows by
    ``company_name`` only.
    """
    logger.info("enter parallel_startup_eval")
    try:
        startups = state.get("startups") or []
        if not startups:
            logger.info("exit parallel_startup_eval (no startups)")
            return {}

        sem = asyncio.Semaphore(config.MAX_PARALLEL_STARTUP_EVALS)

        async def _run_with_retry(kind: str, raw: dict, fn):
            attempts = config.OPENAI_RETRY_MAX_ATTEMPTS
            base = config.OPENAI_RETRY_BASE_SECONDS
            for attempt in range(1, attempts + 1):
                try:
                    async with sem:
                        return await asyncio.to_thread(fn, state, raw)
                except Exception as exc:  # noqa: BLE001
                    if attempt >= attempts or not _is_rate_limit_error(exc):
                        raise
                    sleep_s = base * (2 ** (attempt - 1))
                    logger.warning(
                        "Rate limit in %s eval (%s) attempt %s/%s; retry in %.1fs",
                        kind,
                        raw.get("company_name", "unknown"),
                        attempt,
                        attempts,
                        sleep_s,
                    )
                    await asyncio.sleep(sleep_s)
            raise RuntimeError("unreachable")

        tech_tasks = [_run_with_retry("tech", raw, _run_tech_eval_for_startup) for raw in startups]
        market_tasks = [_run_with_retry("market", raw, _run_market_eval_for_startup) for raw in startups]

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
