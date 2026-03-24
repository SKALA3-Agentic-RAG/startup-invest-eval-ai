from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a market due-diligence analyst for Robotics/AI VC. "
        "Return ONLY JSON compatible with MarketEval schema: "
        "{{company_name, tam_usd_bn, growth_rate_pct, traction_score(0-10), competition_level, rationale}}. "
        "No markdown."
    ),
    (
        "human",
        "거시 맥락:\n{macro_context}\n\n"
        "스타트업 프로필(JSON):\n{startup_json}\n\n"
        "검토 컨텍스트:\n{context}\n\n"
        "위 정보를 바탕으로 시장 관점 투자 적합도를 평가해 MarketEval JSON으로만 답하세요."
    )
])