"""Startup enrichment + is_startup classification from FAISS snippet + Tavily snippets."""

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You validate whether a company is a startup and enrich structured fields. "
            "Respond ONLY with JSON matching StartupProfile fields: "
            "company_name, founded_year, hq_location, funding_stage, total_funding_usd, "
            "business_summary (<=150 words), tech_summary (<=150 words), is_startup (bool). "
            "Startup criteria: founded < 10 years ago OR funding stage Series C or earlier OR "
            "< 500 employees (use web evidence; if unknown, lean conservative). "
            "No markdown fences.",
        ),
        (
            "human",
            "Company name (seed): {company_name}\n"
            "Vector DB snippet:\n{vector_snippet}\n\n"
            "Web search JSON (list of dicts):\n{web_hits}\n",
        ),
    ]
)
