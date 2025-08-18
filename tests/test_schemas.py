import pytest
from app.schemas import ResearchPlan, SourceSummary, FinalBrief, Section

def test_plan_validation():
    plan = ResearchPlan(steps=["step1"])
    assert len(plan.steps) == 1

def test_source_summary_validation():
    summary = SourceSummary(source_url="http://example.com", key_points=["point"], relevance=0.5)
    assert summary.relevance == 0.5

def test_final_brief_validation():
    section = Section(title="Intro", content="Content")
    brief = FinalBrief(topic="Test", summary="Sum", sections=[section], references=[])
    assert len(brief.sections) == 1