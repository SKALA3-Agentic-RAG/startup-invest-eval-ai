PROMPT = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a senior industry analyst at a global market research firm. "
        "Your task is to analyze market size, growth potential, and demand trends based on provided reports and news. "
        "Respond ONLY with a single JSON object. No markdown fences, no prose. "
        "If information is missing from the context, the values should reflect that."
    ),
    (
        "human", 
        "Use ONLY the provided [Context] to answer the question in Korean with a professional, data-driven tone.\n\n"
        "### Instructions:\n"
        "1. **Market Size Analysis (TAM/SAM/SOM)**: Present figures for Total Addressable Market, Serviceable Addressable Market, and Serviceable Obtainable Market.\n"
        "2. **Growth Potential (CAGR)**: Analyze the Compound Annual Growth Rate for the next 5-10 years and identify key market drivers.\n"
        "3. **Demand & News Trend Analysis**: Identify real-world demand (pain points) and capture recent industry shifts or regulatory changes from news data.\n"
        "4. **Citations**: Ensure all claims are backed by specific reports or news citations (Title, Organization, Date).\n\n"
        "If no information is found, strictly say: '죄송하지만 제공된 문서에서는 시장 규모 및 수요에 대한 충분한 정보를 찾을 수 없습니다.'\n\n"
        "### Required JSON Schema:\n"
        "{{\n"
        '  "market_size": "TAM/SAM/SOM analysis in Korean with specific figures",\n'
        '  "growth_potential": "CAGR and market drivers in Korean",\n'
        '  "demand_trends": "Analysis of pain points and news trends in Korean",\n'
        '  "sources": ["Report/Article Title (Organization, YYYY-MM-DD)"]\n'
        "}}\n\n"
        "### Context:\n{context}\n\n"
        "### Question:\n{question}"
    )
])