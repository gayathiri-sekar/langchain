from typing import List
import wikipedia
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Tools Server")

@mcp.tool() #This enables the python function to be a MCP Tool
# Python function - uses wikipedia library to get search results using wikipedia library
def wikipedia_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Python function - uses ddgs library to get search results using DDGS
@mcp.tool() #This enables the python function to be a MCP Tool
def ddg_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            return "\n".join([r["body"] for r in results])
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

# when we start the server, we start in http mode and expose these 2 tools
# when we run this py file, MCP server will start running in localhost:8000/mcp
# Above 2 tools will be exposed for client to use
# transport can be streamable-http or stdio - when MCP server starts