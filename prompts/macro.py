"""Macro environment prompt."""

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a global macro strategist for Robotics/AI investment. "
            "Return ONLY JSON compatible with MacroAnalysis schema: "
            '{{"macro_context": "<korean summary>"}} '
            "(single key only, no markdown).",
        ),
        (
            "human",
            "사용자 투자 질의:\n{query}\n\n"
            "위 질의 관점에서 거시경제/산업/정책 리스크와 기회를 한국어로 요약하고 "
            "반드시 macro_context 단일 필드 JSON으로 답변하세요.",
        ),
    ]
)
