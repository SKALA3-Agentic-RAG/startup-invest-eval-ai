"""Market evaluation prompt."""

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You evaluate market size and traction. "
            "Respond ONLY with JSON for MarketEval: company_name, tam_usd_bn (float), "
            "growth_rate_pct (float estimate), traction_score (0-10), "
            "competition_level (Low|Medium|High), rationale. "
            "No markdown fences.",
        ),
        (
            "human",
            "Macro context:\n{macro_context}\n\n"
            "Startup profile JSON:\n{startup_json}\n\n"
            "Retrieved context:\n{context}\n",
        ),
    ]
)
