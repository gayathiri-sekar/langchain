import asyncio
import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({
    "tools": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    }
})

tools = asyncio.run(client.get_tools())
load_dotenv()
llm = ChatOpenAI(model="gpt-4o")
agent = create_agent(llm, tools)

st.title("AI Agent (MCP Version)")
task = st.text_input("Assign me a task")

if task:
    response = asyncio.run(agent.ainvoke({"messages": task}))
    st.write(response)
    final_output = response["messages"][-1].content
    st.write(final_output)

