import os
import requests
import logging
import json
from typing import List
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from .state import AppState
from .schemas import ResearchPlan, SourceSummary, FinalBrief, Section
from .history import load_history, save_brief
from .tools import search_wrapper
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLMs
llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
llm_hf = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation",
    temperature=0,
    max_new_tokens=256
)

def context_summarization(state: AppState) -> AppState:
    if not isinstance(state, dict):
        state = state.copy() if hasattr(state, 'copy') else dict(state)
    if "context_summary" not in state:
        state["context_summary"] = ""
    if not state.get("follow_up", False):
        return state
    prior_briefs = load_history(state["user_id"])
    logger.debug(f"Prior briefs for user_id {state['user_id']}: {prior_briefs}")
    prompt = ChatPromptTemplate.from_template("Summarize prior briefs: {priors}")
    chain = prompt | llm_gemini | (lambda x: {"context_summary": x.content})
    try:
        result = with_retry(chain, {"priors": prior_briefs})
        state["context_summary"] = result["context_summary"]
        logger.info(f"Context summary generated: {state['context_summary'][:100]}...")
    except Exception as e:
        logger.warning(f"Context summarization failed: {e}. Using empty summary.")
        state["context_summary"] = ""
    return state

def planning(state: AppState) -> AppState:
    context_str = f"Based on prior context: {state['context_summary']}" if state.get("follow_up", False) else ""
    prompt = ChatPromptTemplate.from_template(
        """Create a detailed research plan for the topic: {topic}. {context}
The plan should consist of 3-5 specific search steps or queries to gather information."""
    )
    structured_llm = llm_gemini.with_structured_output(ResearchPlan)
    chain = prompt | structured_llm
    try:
        plan = with_retry(chain, {"topic": state["topic"], "context": context_str})
        state["plan"] = plan
        logger.info(f"Generated research plan: {plan.steps}")
    except Exception as e:
        raw_input = {"topic": state["topic"], "context": context_str}
        logger.error(f"Planning failed: {e}. Raw output: {llm_gemini.invoke(prompt.format_messages(**raw_input)).content}")
        state["plan"] = ResearchPlan(steps=[f"Search for recent information on {state['topic']}"])
    return state

def search(state: AppState) -> AppState:
    sources = []
    max_queries = 3
    for step in state["plan"].steps[:max_queries]:
        if not isinstance(step, str):
            logger.warning(f"Invalid search step type: {type(step)} for step: {step}")
            continue
        try:
            logger.debug(f"Executing search for query: {step}")
            results = search_wrapper(step)
            if not isinstance(results, list):
                logger.warning(f"Search tool returned non-list results for query '{step}': type={type(results)}, value={results}")
                continue
            for result in results:
                if isinstance(result, dict) and "url" in result:
                    sources.append(result["url"])
                elif isinstance(result, dict) and "content" in result:
                    logger.warning(f"No URL found for result with content: {result['content'][:50]}... Using placeholder.")
                    sources.append(f"https://placeholder-{hash(result['content'])}.com")
                else:
                    logger.warning(f"Invalid result format: {result}")
        except Exception as e:
            logger.warning(f"Search failed for step '{step}': {e}")
    state["sources"] = sources
    logger.info(f"Search node completed. Sources found: {len(sources)} - {sources}")
    return state

def content_fetching(state: AppState) -> AppState:
    contents = []
    for url in state["sources"]:
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.get_text(separator='\n', strip=True)
                contents.append(content)
                logger.debug(f"Fetched content from {url}: {content[:100]}...")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == 2:
                    contents.append(f"Fetch error: {str(e)}")
    state["contents"] = contents
    logger.info(f"Content fetching completed. Contents retrieved: {len(contents)}")
    return state

def per_source_summarization(state: AppState) -> AppState:
    prompt = ChatPromptTemplate.from_template(
        """Assess the relevance of the content from {url} to the topic '{topic}' on a scale of 0 to 1.
{content}"""
    )
    structured_llm = llm_gemini.with_structured_output(SourceSummary)
    chain = prompt | structured_llm
    summaries = []
    for url, content in zip(state["sources"], state["contents"]):
        if "error" in content.lower() or len(content.strip()) == 0:
            summaries.append(SourceSummary(source_url=url, relevance=0.1))
            logger.debug(f"Fallback summary for {url}: relevance=0.1")
            continue
        try:
            truncated_content = content[:4000]
            summary = with_retry(chain, {"url": url, "content": truncated_content, "topic": state["topic"]})
            if summary is not None:
                summaries.append(summary)
                logger.debug(f"Generated summary for {url}: relevance={summary.relevance}")
            else:
                logger.warning(f"LLM returned None for {url}. Using fallback.")
                summaries.append(SourceSummary(source_url=url, relevance=0.5))
        except Exception as e:
            logger.warning(f"Summarization failed for {url}: {e}")
            summaries.append(SourceSummary(source_url=url, relevance=0.5))
    if not summaries:
        logger.warning("No summaries generated. Adding fallback summary.")
        summaries.append(SourceSummary(
            source_url="https://example.com",
            relevance=0.3
        ))
    valid_summaries = [s for s in summaries if s is not None]
    logger.info(f"Generated {len(valid_summaries)} valid source summaries: {[s.source_url for s in valid_summaries]}")
    state["summaries"] = valid_summaries
    return state

def synthesis(state: AppState) -> AppState:
    logger.debug("Entering synthesis node")
    prompt = ChatPromptTemplate.from_template(
        """Synthesize a comprehensive brief on the topic '{topic}' based on the following source summaries:
{summaries}
Integrate key points logically, resolve any contradictions, and provide a cohesive narrative.
If no summaries are available, provide a brief overview based on general knowledge of the topic."""
    )
    chain = prompt | llm_gemini | StrOutputParser()
    try:
        summaries_json = json.dumps([s.model_dump() for s in state["summaries"] or []])
        if not state["summaries"]:
            logger.warning("No summaries available for synthesis. Using general overview.")
            summaries_json = "No source summaries available."
        synthesized = with_retry(chain, {"topic": state["topic"], "summaries": summaries_json})
        state["synthesized_brief"] = synthesized
        logger.debug(f"Synthesized brief: {synthesized[:100]}...")
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        state["synthesized_brief"] = f"Failed to synthesize brief for {state['topic']}. No source information available."
    logger.info("Synthesis completed")
    return state

class PartialFinalBrief(BaseModel):
    summary: str
    sections: List[Section]

def post_processing(state: AppState) -> AppState:
    logger.debug("Entering post_processing node")
    logger.info(f"State summaries before post-processing: {len(state['summaries'])} - {[s.source_url for s in state['summaries']]}")
    prompt = ChatPromptTemplate.from_template(
        """Format the synthesized content into a structured research brief on the topic '{topic}'.
Synthesized content: {synth}
Include a high-level summary, detailed sections with titles and content."""
    )
    structured_llm = llm_gemini.with_structured_output(PartialFinalBrief)
    chain = prompt | structured_llm
    try:
        partial = with_retry(chain, {"topic": state["topic"], "synth": state["synthesized_brief"] or "No synthesis available"})
        final = FinalBrief(
            topic=state["topic"],
            summary=partial.summary,
            sections=partial.sections,
            references=state["summaries"]
        )
        state["final_brief"] = final
        save_brief(state["user_id"], final.model_dump_json())
        logger.info(f"Final brief generated with {len(final.references)} references")
    except Exception as e:
        raw_input = {"topic": state["topic"], "synth": state["synthesized_brief"] or "No synthesis available"}
        logger.error(f"Post-processing failed: {e}. Raw output: {llm_gemini.invoke(prompt.format_messages(**raw_input)).content}")
        raise ValueError("Failed to generate final brief")
    logger.info("Post-processing completed")
    return state

def with_retry(chain, inputs, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = chain.invoke(inputs)
            return result
        except (OutputParserException, ValidationError) as e:
            logger.warning(f"Error on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
    logger.error(f"Max retries reached. Returning None.")
    return None