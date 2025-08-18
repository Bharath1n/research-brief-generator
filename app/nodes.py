import os
import requests
import logging
import json
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, AIMessage
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch import cuda
from .state import AppState
from .schemas import ResearchPlan, SourceSummary, FinalBrief
from .history import load_history, save_brief
from .tools import search_tool
from dotenv import load_dotenv
from pydantic import parse as parser

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLMs
llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# Initialize Hugging Face BART model for summarization
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
hf_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
if cuda.is_available():
    hf_model = hf_model.to('cuda')  # Move to GPU if available

def context_summarization(state: AppState) -> AppState:
    if not isinstance(state, dict):
        state = state.copy() if hasattr(state, 'copy') else dict(state)
    if "context_summary" not in state:
        state["context_summary"] = ""
    if not state.get("follow_up", False):
        return state
    prior_briefs = load_history(state["user_id"])
    prompt = ChatPromptTemplate.from_template("Summarize prior briefs: {priors}")
    chain = prompt | llm_gemini | (lambda x: {"context_summary": x.content})
    try:
        result = with_retry(chain, {"priors": prior_briefs})
        state["context_summary"] = result["context_summary"]
    except Exception as e:
        logger.warning(f"Context summarization failed: {e}. Using empty summary.")
        state["context_summary"] = ""
    return state

def hf_summarize(url, content):
    if not content or "Fetch error" in content:
        return parser.parse({"source_url": url, "key_points": ["No content available"], "relevance": 0.1})
    inputs = tokenizer(f"Summarize: {content}", return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
    outputs = hf_model.generate(**inputs, max_length=150, min_length=40, no_repeat_ngram_size=2)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Summary for {url}: {summary}")
    return parser.parse({"source_url": url, "key_points": [summary], "relevance": 0.9})

def planning(state: AppState) -> AppState:
    parser = PydanticOutputParser(pydantic_object=ResearchPlan)
    prompt = ChatPromptTemplate.from_template(
        "Plan research for {topic}. Depth: {depth}. Context: {context}. {format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm_gemini | parser
    state["plan"] = with_retry(chain, {"topic": state["topic"], "depth": state["depth"], "context": state["context_summary"] or ""})
    return state

def search(state: AppState) -> AppState:
    try:
        query = f"recent articles on {state['topic']}"
        results = search_tool.invoke({"query": query, "max_results": state["depth"] * 3})
        logger.info(f"Raw Tavily response: {results}")
        if isinstance(results, str):
            results = json.loads(results)
        state["sources"] = [r["url"] for r in results.get('results', []) if isinstance(r, dict) and "url" in r]
        logger.info(f"Found {len(state['sources'])} sources: {state['sources']}")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        state["sources"] = []
    return state

def content_fetching(state: AppState) -> AppState:
    contents = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    for url in state["sources"]:
        try:
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            # Try lxml first, fall back to html.parser
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception as e:
                logger.warning(f"lxml parser failed for {url}: {e}. Falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            content = ' '.join(p.get_text() for p in soup.find_all(['p', 'div']) if p.get_text().strip())[:2000]
            if not content:
                content = "No usable content found, possibly JavaScript-rendered."
            contents.append(content)
            logger.info(f"Fetched content length from {url}: {len(content)}")
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            contents.append(f"Fetch error for {url}: {str(e)}")
    state["contents"] = contents
    return state

def per_source_summarization(state: AppState) -> AppState:
    parser = PydanticOutputParser(pydantic_object=SourceSummary)
    prompt = ChatPromptTemplate.from_template(
        "Summarize source {url}: {content}. {format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    def hf_summarize(url, content):
        if not content or "Fetch error" in content:
            error_msg = content if "Fetch error" in content else "No content available"
            logger.warning(f"Skipping summarization for {url}: {error_msg}")
            return parser.parse({"source_url": url, "key_points": [error_msg], "relevance": 0.1})
        inputs = tokenizer(f"Summarize: {content}", return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        outputs = hf_model.generate(**inputs, max_length=150, min_length=40, no_repeat_ngram_size=2)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated summary for {url}: {summary[:100]}...")  # Log first 100 chars
        return parser.parse({"source_url": url, "key_points": [summary], "relevance": 0.9})

    summaries = []
    for url, content in zip(state["sources"], state["contents"]):
        try:
            summary = hf_summarize(url, content)
            summaries.append(summary)
        except Exception as e:
            logger.warning(f"HF summarization failed for {url}: {e}")
            chain = prompt | llm_gemini | parser
            summaries.append(with_retry(chain, {"url": url, "content": content or "No content available"}))
    state["summaries"] = summaries
    logger.info(f"Generated {len(summaries)} summaries")
    return state

def synthesis(state: AppState) -> AppState:
    prompt = ChatPromptTemplate.from_template("Synthesize a brief on {topic} based on the following summaries: {summaries}")
    chain = prompt | llm_gemini
    state["synthesized_brief"] = with_retry(chain, {"topic": state["topic"], "summaries": state["summaries"] or []}).content
    return state

def post_processing(state: AppState) -> AppState:
    parser = PydanticOutputParser(pydantic_object=FinalBrief)
    prompt = ChatPromptTemplate.from_template(
        "Format a final brief on {topic} with the following synthesized content: {synth}. Ensure the summary and sections are relevant to the topic. {format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm_gemini | parser
    final = with_retry(chain, {"topic": state["topic"], "synth": state["synthesized_brief"] or "No synthesis available"})
    final.topic = state["topic"]
    # Convert SourceSummary objects to dictionaries for compatibility
    final.references = [summary.model_dump() for summary in state["summaries"]]
    state["final_brief"] = final
    save_brief(state["user_id"], final.model_dump_json())
    return state

def with_retry(chain, inputs, max_retries=3):
    for _ in range(max_retries):
        try:
            return chain.invoke(inputs)
        except OutputParserException:
            pass
    raise ValueError("Failed after retries")