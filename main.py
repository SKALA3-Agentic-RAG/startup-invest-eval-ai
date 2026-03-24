"""CLI entry: stream LangGraph run and persist the Markdown report."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import datetime, timezone

import config as app_config
from agents.graph import build_graph
from memory.checkpointer import async_checkpointer
from tools.ingest_pdfs import ingest_from_folder


async def _run_async(lg_config: dict, initial_state: dict) -> dict:
    """
    Run the graph under an async SQLite checkpointer and return final state values.

    Uses ``astream`` because several nodes are ``async def`` (parallel eval + search).
    """
    async with async_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        async for step in graph.astream(initial_state, config=lg_config, stream_mode="updates"):
            for node_name, payload in step.items():
                print(f"\n=== node: {node_name} ===")
                print(payload)
        snap = await graph.aget_state(lg_config)
        return snap.values or {}


def main() -> None:
    """Parse args, run the graph with checkpointing, and save the report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="Agentic RAG startup investment workflow")
    parser.add_argument(
        "--query",
        required=True,
        help='Research question, e.g. "AI healthcare startups Korea 2024"',
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="LangGraph thread id for checkpointing (default: random UUID).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip PDF->FAISS ingest before running the graph.",
    )
    parser.add_argument(
        "--ingest-engine",
        choices=("auto", "pdfplumber", "pymupdf"),
        default="auto",
        help="Ingest engine for PDF loading (auto/pdfplumber/pymupdf).",
    )
    args = parser.parse_args()

    if not app_config.OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is not set. Copy .env.example to .env and fill keys.")
        sys.exit(1)
    if not app_config.TAVILY_API_KEY:
        logging.warning("TAVILY_API_KEY missing — web search will return no results.")

    if not args.skip_ingest:
        logging.info(
            "Ingesting PDFs from %s into FAISS (engine=%s)",
            app_config.PDF_SOURCE_DIR,
            args.ingest_engine,
        )
        ingest_code = ingest_from_folder(engine=args.ingest_engine)
        if ingest_code != 0:
            logging.error(
                "PDF ingest failed or no documents found in %s. "
                "Add PDFs then rerun, or use --skip-ingest to run anyway.",
                app_config.PDF_SOURCE_DIR,
            )
            sys.exit(1)

    thread_id = args.thread_id or str(uuid.uuid4())
    lg_config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "query": args.query,
        "macro_context": None,
        "startups": [],
        "current_index": 0,
        "tech_evals": [],
        "market_evals": [],
        "scores": [],
        "invest_decisions": [],
        "final_report": None,
        "error": None,
    }

    logging.info("Starting run (thread_id=%s)", thread_id)
    values = asyncio.run(_run_async(lg_config, initial_state))

    report_md = values.get("final_report")
    err = values.get("error")

    if err:
        logging.warning("Run finished with error flag: %s", err)

    if report_md:
        app_config.REPORT_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = app_config.REPORT_OUTPUT_PATH / f"{ts}.md"
        out_path.write_text(report_md, encoding="utf-8")
        logging.info("Wrote report to %s", out_path)
    else:
        logging.warning("No final_report in graph state; nothing written to output/reports.")


if __name__ == "__main__":
    main()
