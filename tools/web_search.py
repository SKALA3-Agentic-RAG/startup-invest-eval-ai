"""Tavily web search wrapper for LangChain community tool."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, List

import config

logger = logging.getLogger(__name__)


@lru_cache
def _tool():
    """Lazily construct TavilySearchResults with API key from config."""
    from langchain_community.tools.tavily_search import TavilySearchResults

    # TODO: swap to langchain-tavily if you migrate off community tool warnings.
    return TavilySearchResults(
        max_results=5,
        tavily_api_key=config.TAVILY_API_KEY,
    )


def search(query: str) -> List[dict[str, Any]]:
    """Run Tavily search and return a list of result dicts (content, url, ...)."""
    if not config.TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY missing; web search returns empty.")
        return []
    tool = _tool()
    raw = tool.invoke({"query": query})
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [{"content": raw, "url": ""}]
    return []
