from pydantic import BaseModel, Field
from typing import List

class Section(BaseModel):
    title: str
    content: str

class ResearchPlan(BaseModel):
    """Research planning steps."""
    steps: List[str] = Field(..., description="List of steps to research the topic")

class SourceSummary(BaseModel):
    """Summary of an individual source."""
    source_url: str
    key_points: List[str]
    relevance: float = Field(..., ge=0, le=1)  # 0-1 score

class FinalBrief(BaseModel):
    """Compiled research brief."""
    topic: str
    summary: str
    sections: List[Section]
    references: List[SourceSummary]