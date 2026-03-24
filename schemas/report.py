"""Final investment report schemas."""

from typing import List

from pydantic import BaseModel, Field


class InvestDecision(BaseModel):
    """Per-startup investment stance."""

    company_name: str
    tech_score: float = Field(ge=0, le=10, description="Technical score derived in decision stage")
    market_score: float = Field(ge=0, le=10, description="Market score derived in decision stage")
    total_score: float = Field(ge=0, le=10, description="Weighted total score")
    rank: int = Field(ge=1, description="Rank among candidates by total_score (1 is best)")
    decision: str = Field(description='One of "GO", "WATCH", "PASS"')
    rationale: str
    key_risks: List[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    """Structured report payload including Markdown body."""

    title: str
    generated_at: str = Field(description="ISO-8601 timestamp")
    macro_summary: str
    startup_count: int
    decisions: List[InvestDecision]
    top_pick: str
    full_report_md: str = Field(description="Complete Markdown investment memo")


class InvestDecisionsBatch(BaseModel):
    """Wrapper for structured LLM output of multiple decisions."""

    decisions: List[InvestDecision]
