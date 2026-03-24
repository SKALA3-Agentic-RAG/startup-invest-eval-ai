"""Final Markdown report prompt."""

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You produce institutional-grade investment memos. "
            "Respond ONLY with JSON for FinalReport: title, generated_at (ISO-8601 string), "
            "macro_summary, startup_count (int), decisions (same objects as input), "
            "top_pick (company name), full_report_md (complete Markdown report with sections: "
            "Executive Summary, Macro, Methodology, Company Deep Dives, Scoring Table, "
            "Decisions, Risks, Appendix). "
            "No markdown fences around the JSON.",
        ),
        (
            "human",
            "Original query:\n{query}\n\n"
            "Decisions JSON:\n{decisions_json}\n\n"
            "Scores JSON:\n{scores_json}\n\n"
            "Startups JSON:\n{startups_json}\n",
        ),
    ]
)
