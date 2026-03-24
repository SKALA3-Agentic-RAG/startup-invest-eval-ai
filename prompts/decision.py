"""Investment decision batch prompt."""

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an investment committee member. "
            'Respond ONLY with JSON: {{"decisions": [InvestDecision, ...]}} where each '
            "InvestDecision has company_name, decision (GO|WATCH|PASS), rationale, key_risks (string array). "
            "No markdown fences.",
        ),
        (
            "human",
            "Macro summary:\n{macro_context}\n\n"
            "Scores JSON:\n{scores_json}\n\n"
            "Tech evaluations JSON:\n{tech_json}\n\n"
            "Market evaluations JSON:\n{market_json}\n",
        ),
    ]
)
