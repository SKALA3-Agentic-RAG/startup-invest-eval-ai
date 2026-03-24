from langchain_core.prompts import ChatPromptTemplate

# Technical Evaluation Prompt (Synchronized with team code)
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are an expert technical due diligence analyst. "
        "Evaluate the startup's technology based on macro trends, company profile, and provided context. "
        "Output the results in Korean within the structured format."
    ),
    (
        "human", 
        "### Input Data:\n"
        "1. **Macro Context**: {macro_context}\n"
        "2. **Startup Profile**: {startup_json}\n"
        "3. **Research Context**: {context}\n\n"
        "### Instructions:\n"
        "1. **Core Technology Stack**: Analyze the hardware/software mentioned in the Startup Profile and Research Context.\n"
        "2. **Technical Pros & Cons**: Evaluate the approach considering the current Macro Context (e.g., labor shortages, interest rates).\n"
        "3. **Moat & Scalability**: Assess patents, data barriers, and mass production readiness.\n"
        "4. **Competitors**: Identify and compare with similar startups.\n"
        "5. **Citations**: Must include specific sources from the Research Context.\n\n"
        "Write the analysis in a professional, data-driven Korean tone.\n"
        "If data is insufficient, state: '제공된 문서에서 관련 기술 정보를 찾을 수 없습니다.'"
    )
])