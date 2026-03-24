"""SQLite checkpointer for LangGraph (async API for graphs with ``async def`` nodes)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import aiosqlite
import config

if TYPE_CHECKING:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)


def _patch_aiosqlite_connection_is_alive() -> None:
    """
    ``langgraph-checkpoint-sqlite`` 2.0.x calls ``conn.is_alive()`` in ``AsyncSqliteSaver.setup``,
    but ``aiosqlite.Connection`` has never implemented that method, which raises
    ``AttributeError``. Newer LangGraph uses a fallback (see ``_build_conn_started_check`` on
    main). We mirror that here so 2.0.11 works with current ``aiosqlite``.

    TODO: Drop this when upgrading to ``langgraph-checkpoint-sqlite`` 3.x+ (includes the fix).
    """
    if callable(getattr(aiosqlite.Connection, "is_alive", None)):
        return

    def is_alive(self: aiosqlite.Connection) -> bool:
        thread = getattr(self, "_thread", None)
        return False if thread is None else thread.is_alive()

    aiosqlite.Connection.is_alive = is_alive  # type: ignore[method-assign]


_patch_aiosqlite_connection_is_alive()


@asynccontextmanager
async def async_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """
    Yield an :class:`~langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver` for ``compile``.

    Must be used with ``graph.astream`` / ``graph.ainvoke`` (sync ``SqliteSaver`` cannot
    back async runs). Closes the DB when the context exits.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    config.CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    path = str(config.CHECKPOINT_DB_PATH)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        logger.info("AsyncSqliteSaver ready at %s", path)
        yield saver
