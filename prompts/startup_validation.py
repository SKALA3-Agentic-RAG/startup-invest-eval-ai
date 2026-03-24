"""Startup enrichment + is_startup classification from FAISS snippet + Tavily snippets."""

from langchain_core.prompts import ChatPromptTemplate
CITATION_FORMAT_GUIDE_KR = """
**출처 표기 예시**
* 기관 보고서: 한국은행(2024). *금융안정보고서*. https://www.bok.or.kr/...
* 학술 논문: 김철수(2024). *인공지능 산업 전망*. 투자연구, 10(2), 50-60.
* 웹페이지: IEA(2024). *Global EV Outlook 2024*. IEA. https://...
"""



PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""You are an expert assistant for Venture Capital Startup Investment Analysis.
            Use the following pieces of retrieved context to answer the question accurately and comprehensively.

            Instructions:
            - use information from the provided context and web search results to answer the question
            - If you don't know the answer based on the context, say "죄송하지만 제공된 문서에서는 해당 정보를 찾을 수 없습니다."
            - Answer in Korean with a clear and professional tone
            - Cite specific parts of the context when possible
            """ + CITATION_FORMAT_GUIDE_KR + """
            - Define the concept of startup
            #Answer as a json format
            """
        ),
        (
            "human",
            "Company name (seed): {company_name}\n"
            "Vector DB snippet:\n{vector_snippet}\n\n"
            "Web search JSON (list of dicts):\n{web_hits}\n",
        ),
    ]
)
