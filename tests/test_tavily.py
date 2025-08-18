from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
load_dotenv()
search_tool = TavilySearch(max_results=5)
results = search_tool.invoke({"query": "AI Ethics", "max_results": 15})
print(results)