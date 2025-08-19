Context-Aware Research Brief Generator
A research assistant that generates structured, evidence-linked briefs with follow-up support using LangGraph and LangChain. Features a deployed API, CLI, and optional Streamlit UI.

Deployment: https://research-brief-generator-5c5c.onrender.com
Repo: https://github.com/Bharath1n/research-brief-generator

Problem Statement and Objective
Creates research briefs from user topics with web evidence, supporting follow-up queries with context. Aims to use LangGraph for orchestration, LangChain for abstraction, and Pydantic for schema validation, with a deployed API and CLI.

Graph Architecture
Uses a StateGraph with 7 nodes for processing.

Visual Representation
mermaidgraph TD
    A[API/CLI] --> B[Context Summarization]
    A --> C[Planning]
    B --> C
    C --> D[Search (Tavily)]
    D --> E[Content Fetching (BSoup)]
    E --> F[Per-Source Summ (Gemini)]
    F --> G[Synthesis (Gemini)]
    G --> H[Post-Processing]
    H --> I[FinalBrief JSON]
    I --> J[Streamlit (Local)]
    I --> K[SQLite History]
    L[Checkpointing] --> B & C & D & E & F & G & H

Nodes: Context Summarization, Planning, Search, Content Fetching, Per-Source Summarization, Synthesis, Post-Processing.
Flow: Conditional follow_up entry, checkpointing with MemorySaver.
Note: Export this Mermaid code at mermaid.live as PNG and replace with ![Diagram](diagram.png) after uploading.

Model and Tool Selection

Gemini (gemini-1.5-flash): Fast, cost-effective LLM for summarization and synthesis.
Tavily: Efficient web search with relevance filtering.
BeautifulSoup: Reliable content parsing.
Rationale: Optimized for cost and speed; HuggingFace removed to fix Vercel OOM.

Schema Definitions and Validation

Schemas (schemas.py):

ResearchPlan: List[str] for steps.
SourceSummary: source_url: str, relevance: float (0-1).
FinalBrief: topic: str, summary: str, sections: List[Section], references: List[SourceSummary].
Validation: Pydantic with with_structured_output, API input checks, and with_retry (3 attempts) for errors.

Deployment Instructions
Prerequisites
Python 3.9+
Git
API keys (Google, Tavily)

Local

Clone: git clone https://github.com/Bharath1n/research-brief-generator
Install: pip install -r requirements.txt
Set .env from .env.example with API keys.
Run API: uvicorn app.api:api_app --reload
Run UI: streamlit run frontend.py
Use make (optional): make install, make run.

Render

Sign up at render.com.
New Web Service, connect repo.
Set: Python runtime, pip install -r requirements.txt, uvicorn app.api:api_app --host 0.0.0.0 --port $PORT.
Add env vars.
Deploy (5-10 min).

Streamlit
Run: streamlit run frontend.py, enter params, click "Generate Brief".

Cost and Latency Benchmarks

Latency: 5-15s (depth=3, local/Render with cold starts).
Tokens: 3000-6000/run.
Cost: ~$0.01-0.03 (Gemini) + ~$0.001 (Tavily).
Tracing: LangSmith enabled, [insert trace link/screenshot].

Limitations and Areas for Improvement

Limits: API rate caps, no multi-modal, SQLite scalability issues, free tier cold starts.
Improvements: Async fetching, multi-LLM, token tracking, enhanced tests, scalable DB.