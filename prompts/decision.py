"""Investment decision batch prompt."""

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a VC investment committee associate.
            Use macro, tech analysis context, and market analysis context to produce
            final investment decisions.

            Return ONLY JSON compatible with InvestDecisionsBatch:
            {{
              "decisions": [
                {{
                  "company_name": "...",
                  "tech_score": 0-10,
                  "market_score": 0-10,
                  "total_score": 0-10,
                  "rank": 1+,
                  "decision": "GO|WATCH|PASS",
                  "rationale": "...",
                  "key_risks": ["...", "..."]
                }}
              ]
            }}

            Scoring rule:
            total_score = ({score_weight_tech} * tech_score) + ({score_weight_market} * market_score)
            Decision guide:
            - GO: total_score >= 8.0
            - WATCH: 6.0 <= total_score < 8.0
            - PASS: total_score < 6.0

            Never invent evidence; if context is weak, lower score and explain uncertainty.
            """
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
