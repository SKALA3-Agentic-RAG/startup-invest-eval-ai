"""SQLite-backed LangGraph checkpointer (persistent thread state)."""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

import config

logger = logging.getLogger(__name__)

_saver: Optional[object] = None


def get_checkpointer():
    """
    Return a process-singleton :class:`SqliteSaver`.

    Uses ``config.CHECKPOINT_DB_PATH``; creates parent directories as needed.
    """
    global _saver
    if _saver is not None:
        return _saver

    from langgraph.checkpoint.sqlite import SqliteSaver

    config.CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.CHECKPOINT_DB_PATH), check_same_thread=False)
    _saver = SqliteSaver(conn)
    logger.info("SqliteSaver ready at %s", config.CHECKPOINT_DB_PATH)
    return _saver
