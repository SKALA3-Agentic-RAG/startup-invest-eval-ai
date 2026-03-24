"""Final investment report schemas."""

from typing import List

from pydantic import BaseModel, Field


class InvestDecision(BaseModel):
    """Per-startup investment stance."""

    company_name: str
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
