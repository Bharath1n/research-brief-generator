import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
import logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Tavily API Key loaded: {os.getenv('TAVILY_API_KEY')}")
try:
    search_tool = TavilySearch(max_results=5)
    logger.info("Tavily search tool initialized successfully")
except Exception as e:
    logger.error(f"Tavily initialization failed: {e}")