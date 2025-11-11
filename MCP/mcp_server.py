from typing import List
import wikipedia
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP


def wikipedia_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Error: {str(e)}"


def ddg_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            return "\n".join([r["body"] for r in results])
    except Exception as e:
        return f"Error: {str(e)}"
