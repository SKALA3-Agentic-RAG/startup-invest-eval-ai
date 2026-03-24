# prompts/macro.py
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a global macro-economic strategist. Evaluate how macro indicators, "
        "government policies, and global trends impact the robotics industry. "
        "Output the results into the structured format provided."
    ),
    (
        "human", 
        "Analyze the following research query in Korean: {query}\n\n"
        "### Instructions:\n"
        "1. Macro-Economic Environment (Rates, Inflation)\n"
        "2. Industrial & Global Trends (Reshoring, Labor shortage)\n"
        "3. Policy & Regulatory Framework (Subsidies, Risks)\n"
        "4. Include specific citations.\n\n"
        "Combine all analysis into the 'macro_context' field."
    )
])