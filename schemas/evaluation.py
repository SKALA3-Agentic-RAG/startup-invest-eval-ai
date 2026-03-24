"""Evaluation and scoring schemas."""

from pydantic import BaseModel, Field


class MacroAnalysis(BaseModel):
    """LLM output for macro / sector trends."""

    macro_context: str


class TechEval(BaseModel):
    """Technical capability assessment for one company."""

    company_name: str
    tech_score: float = Field(ge=0, le=10)
    innovation_level: str = Field(description="Low / Medium / High")
    ip_strength: str
    tech_risk: str
    rationale: str


class MarketEval(BaseModel):
    """Market and traction assessment for one company."""

    company_name: str
    tam_usd_bn: float = Field(description="Estimated TAM in billions USD")
    growth_rate_pct: float = Field(description="Estimated market or company growth %")
    traction_score: float = Field(ge=0, le=10)
    competition_level: str = Field(description="Low / Medium / High")
    rationale: str


class ScoreCard(BaseModel):
    """Weighted score and rank for ranking startups."""

    company_name: str
    tech_score: float
    market_score: float
    total_score: float = Field(description="0.4 * tech + 0.6 * market (0–10 scale inputs)")
    rank: int
