"""Pydantic models for startup profiles."""

from typing import Optional

from pydantic import BaseModel, Field


class StartupProfile(BaseModel):
    """Structured startup record used across search and evaluation."""

    company_name: str = Field(description="Legal or brand name of the company")
    founded_year: Optional[int] = Field(default=None, description="Year founded if known")
    hq_location: Optional[str] = Field(default=None, description="Headquarters city/country")
    funding_stage: Optional[str] = Field(
        default=None,
        description="Seed / Series A / Series B / Series C / etc.",
    )
    total_funding_usd: Optional[float] = Field(default=None, description="Total raised in USD")
    business_summary: str = Field(description="Business model and market; max ~150 words")
    tech_summary: str = Field(description="Core tech stack and innovation; max ~150 words")
    is_startup: bool = Field(description="Whether the entity matches startup criteria")
