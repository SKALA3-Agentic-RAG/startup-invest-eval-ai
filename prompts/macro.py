"""Macro environment prompt."""

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior equity research analyst. Respond ONLY with a single JSON object "
            'matching the schema: {{"macro_context": string}}. No markdown fences, no prose.',
        ),
        (
            "human",
            "User investment research query:\n{query}\n\n"
            "Summarize macro conditions and AI industry / sector trends relevant to this query "
            "(rates, regulation, adoption, capex cycles). Be concise but substantive.",
        ),
    ]
)
