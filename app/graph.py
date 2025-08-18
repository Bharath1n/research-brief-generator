from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AppState
from .nodes import (
    context_summarization, planning, search, content_fetching,
    per_source_summarization, synthesis, post_processing
)

graph = StateGraph(state_schema=AppState)

graph.add_node("context_summarization", context_summarization)
graph.add_node("planning", planning)
graph.add_node("search", search)
graph.add_node("content_fetching", content_fetching)
graph.add_node("per_source_summarization", per_source_summarization)
graph.add_node("synthesis", synthesis)
graph.add_node("post_processing", post_processing)

# Conditional entry point
graph.set_conditional_entry_point(
    lambda state: "context_summarization" if state.get("follow_up", False) else "planning",
    {
        "context_summarization": "context_summarization",
        "planning": "planning"
    }
)
graph.add_edge("context_summarization", "planning")
graph.add_edge("planning", "search")
graph.add_edge("search", "content_fetching")
graph.add_edge("content_fetching", "per_source_summarization")
graph.add_edge("per_source_summarization", "synthesis")
graph.add_edge("synthesis", "post_processing")
graph.add_edge("post_processing", END)

# Checkpointing
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)