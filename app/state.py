# In state.py
from typing import TypedDict, Annotated, List
from langgraph.graph import add_messages
from .schemas import ResearchPlan, SourceSummary, FinalBrief

class AppState(TypedDict):
    topic: str
    depth: int
    user_id: str
    follow_up: bool
    context_summary: str  # Summarized prior interactions
    plan: ResearchPlan
    sources: List[str]  # URLs from search
    contents: List[str]  # Fetched content from URLs
    summaries: List[SourceSummary]  # Per-source summaries (remove add_messages)
    synthesized_brief: str  # Pre-final synthesis
    final_brief: FinalBrief