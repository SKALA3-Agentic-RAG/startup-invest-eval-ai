"""Technical evaluation prompt."""

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You assess deep technical capability for AI startups. "
            "Respond ONLY with JSON for TechEval: company_name, tech_score (0-10 float), "
            "innovation_level (Low|Medium|High), ip_strength, tech_risk, rationale. "
            "No markdown fences.",
        ),
        (
            "human",
            "Macro context:\n{macro_context}\n\n"
            "Startup profile JSON:\n{startup_json}\n\n"
            "Retrieved context (may include vector DB + web):\n{context}\n",
        ),
    ]
)
