# Context-Aware Research Brief Generator

A research assistant that generates structured, evidence-linked briefs with follow-up support using **LangGraph** and **LangChain**.  
Features a deployed API, CLI, and optional Streamlit UI.

- **Deployment**: [Live App](https://research-brief-generator-5c5c.onrender.com)  
- **Repository**: [GitHub Repo](https://github.com/Bharath1n/research-brief-generator)


## Problem Statement and Objective

This project creates **research briefs** from user topics with web evidence, supporting **follow-up queries with context**.  
It leverages:

- **LangGraph** for orchestration  
- **LangChain** for abstraction  
- **Pydantic** for schema validation  


## System Architecture

The system is designed as a **graph-based workflow** with modular nodes.  

### Mermaid Graph Representation

graph TD
    A[User Input: API/CLI] --> B[Context Summarization<br>Node]
    A --> C[Planning Node]
    B --> C
    C --> D[Search Node<br>(Tavily API)]
    D --> E[Content Fetching Node<br>(Requests, BeautifulSoup)]
    E --> F[Per-Source Summarization Node<br>(Gemini LLM)]
    F --> G[Synthesis Node<br>(Gemini LLM)]
    G --> H[Post-Processing Node]
    H --> I[Output: FinalBrief JSON]
    I --> J[Streamlit UI<br>(Local)]
    I --> K[SQLite History]
    L[Checkpointing<br>(MemorySaver)] --> B
    L --> C
    L --> D
    L --> E
    L --> F
    L --> G
    L --> H


## Node Descriptions

- Context Summarization → Condenses input for planning.
- Planning → Breaks down tasks into structured steps.
- Search (Tavily API) → Web retrieval with relevance filtering.
- Content Fetching → Extracts raw content via requests + BeautifulSoup.
- Per-Source Summarization → Summarizes each source using Gemini LLM.
- Synthesis → Combines summaries into a unified research brief.
- Post-Processing → Cleans, validates, and formats output.
- FinalBrief JSON → Structured output stored in SQLite and viewable via Streamlit.
- Checkpointing (MemorySaver) → Ensures fault tolerance and recoverability.


## Model and Tool Selection

- Gemini (gemini-1.5-flash) → Summarization & synthesis (fast + cost-efficient).
- Tavily → Efficient web search with relevance ranking.
- BeautifulSoup → Reliable content parsing.
- HuggingFace was removed to fix Vercel OOM issues; optimized for cost & speed.


## Schema Definitions (schemas.py)

- ResearchPlan → steps: List[str]
- SourceSummary → source_url: str, relevance: float
- FinalBrief → topic: str, summary: str, sections: List[Section], references: List[SourceSummary]
- Validation via Pydantic + with_structured_output and retry logic (with_retry, 3 attempts).


## Deployment
Prerequisites

- Python 3.9+
- Git
- API Keys (Google, Tavily)


## Local Setup
- git clone https://github.com/Bharath1n/research-brief-generator
- cd research-brief-generator
- pip install -r requirements.txt
- cp .env.example .env   # Add API keys
- uvicorn app.api:api_app --reload
- streamlit run frontend.py


## Render Deployment

- Create a new Web Service on Render
- Connect repo, set runtime = Python, and start command:
- uvicorn app.api:api_app --host 0.0.0.0 --port $PORT
- Add env vars and deploy (takes ~5–10 mins).


## Benchmarks

- Latency → 5–15s (depth=3, local/Render).
- Tokens → 3k–6k per run.
- Cost → ~$0.01–0.03 (Gemini) + ~$0.001 (Tavily).
- Tracing → Enabled via LangSmith.


## Limitations

- API rate limits.
- No multimodal support.
- SQLite not scalable for production.
- Render free tier cold starts.


## Future Improvements

- Async fetching for faster runs.
- Multi-LLM integration.
- Token usage tracking.
- Scalable DB (Postgres).
- End-to-end test coverage.