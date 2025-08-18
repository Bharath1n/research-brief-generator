import pytest
from unittest.mock import patch, AsyncMock
from app.nodes import (
    context_summarization, planning, search, content_fetching,
    per_source_summarization, synthesis, post_processing, with_retry,
    fetch_content
)
from app.state import AppState
from app.schemas import ResearchPlan, SourceSummary, FinalBrief, Section
from app.tools import search_tool

@pytest.fixture
def mock_state():
    return AppState(
        topic="test",
        depth=1,
        follow_up=False,
        user_id="test",
        context_summary="",
        sources=[],
        contents=[],
        summaries=[],
        synthesized_brief="",
        final_brief=None,
        plan=None
    )

@pytest.fixture
def mock_gemini():
    return AsyncMock()

@patch('app.nodes.llm_gemini.invoke', new_callable=AsyncMock)
def test_context_summarization(mock_invoke, mock_state):
    mock_invoke.return_value = "Summarized context"
    result = context_summarization(mock_state)
    assert result["context_summary"] == "Summarized context"
    assert not mock_invoke.called if not mock_state["follow_up"] else mock_invoke.called

@patch('app.nodes.llm_gemini.invoke', new_callable=AsyncMock)
def test_planning(mock_invoke, mock_state):
    mock_invoke.return_value = ResearchPlan(steps=["Step1"])
    result = planning(mock_state)
    assert isinstance(result["plan"], ResearchPlan)
    assert len(result["plan"].steps) == 1

@patch('app.nodes.search_tool.invoke')
def test_search(mock_invoke, mock_state):
    mock_invoke.return_value = [{"url": "http://example.com"}]
    result = search(mock_state)
    assert len(result["sources"]) == 1
    assert result["sources"][0] == "http://example.com"

@patch('app.nodes.fetch_all')
async def test_content_fetching(mock_fetch_all, mock_state):
    mock_state["sources"] = ["http://example.com"]
    mock_fetch_all.return_value = ["Sample content"]
    result = await content_fetching(mock_state)
    assert len(result["contents"]) == 1
    assert result["contents"][0] == "Sample content"

@patch('app.nodes.llm_gemini.invoke', new_callable=AsyncMock)
def test_per_source_summarization(mock_invoke, mock_state):
    mock_state["sources"] = ["http://example.com"]
    mock_state["contents"] = ["Sample content"]
    mock_invoke.return_value = SourceSummary(source_url="http://example.com", key_points=["point"], relevance=0.8)
    result = per_source_summarization(mock_state)
    assert len(result["summaries"]) == 1
    assert isinstance(result["summaries"][0], SourceSummary)

@patch('app.nodes.llm_gemini.invoke', new_callable=AsyncMock)
def test_synthesis(mock_invoke, mock_state):
    mock_state["summaries"] = [SourceSummary(source_url="http://example.com", key_points=["point"], relevance=0.8)]
    mock_invoke.return_value = "Synthesized brief"
    result = synthesis(mock_state)
    assert result["synthesized_brief"] == "Synthesized brief"

@patch('app.nodes.llm_gemini.invoke', new_callable=AsyncMock)
def test_post_processing(mock_invoke, mock_state):
    mock_state["summaries"] = [SourceSummary(source_url="http://example.com", key_points=["point"], relevance=0.8)]
    mock_state["synthesized_brief"] = "Synthesized brief"
    mock_invoke.return_value = FinalBrief(
        topic="test",
        summary="Summary",
        sections=[Section(title="Intro", content="Content")],
        references=[]
    )
    result = post_processing(mock_state)
    assert isinstance(result["final_brief"], FinalBrief)
    assert result["final_brief"].topic == "test"

@patch('app.nodes.llm_gemini.invoke', new_callable=AsyncMock)
def test_with_retry_success(mock_invoke, mock_state):
    mock_invoke.return_value = "Success"
    result = with_retry(lambda x: mock_invoke(x), {"test": "data"})
    assert result == "Success"

@patch('app.nodes.llm_gemini.invoke', side_effect=Exception("Parser Error"))
def test_with_retry_failure(mock_invoke, mock_state):
    with pytest.raises(ValueError):
        with_retry(lambda x: mock_invoke(x), {"test": "data"})

@patch('aiohttp.ClientSession.get')
async def test_fetch_content_success(mock_get, mock_state):
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text.return_value = "<p>Sample</p>"
    mock_get.return_value.__aenter__.return_value = mock_response
    content = await fetch_content("http://example.com", AsyncMock())
    assert "Sample" in content

@patch('app.nodes.llm_gemini.invoke', new_callable=AsyncMock)
@patch('app.nodes.search_tool.invoke')
@patch('app.nodes.fetch_all')
async def test_e2e_graph(mock_fetch_all, mock_search, mock_llm, mock_state):
    mock_search.return_value = [{"url": "http://example.com"}]
    mock_fetch_all.return_value = ["Sample content"]
    mock_llm.side_effect = [
        "Summarized context",  # context_summarization
        ResearchPlan(steps=["Step1"]),  # planning
        SourceSummary(source_url="http://example.com", key_points=["point"], relevance=0.8),  # per_source
        "Synthesized brief",  # synthesis
        FinalBrief(
            topic="test",
            summary="Summary",
            sections=[Section(title="Intro", content="Content")],
            references=[]
        )  # post_processing
    ]
    
    state = mock_state.copy()
    state["follow_up"] = True  # Trigger context summarization
    state = context_summarization(state)
    state = planning(state)
    state = search(state)
    state = await content_fetching(state)
    state = per_source_summarization(state)
    state = synthesis(state)
    state = post_processing(state)
    
    assert isinstance(state["final_brief"], FinalBrief)
    assert state["final_brief"].topic == "test"