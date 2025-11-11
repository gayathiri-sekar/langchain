import asyncio
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


tools = None

llm = ChatOpenAI(model="gpt-4o")
agent = create_agent(llm, tools)

st.title("AI Agent (MCP Version)")
task = st.text_input("Assign me a task")

if task:
    response = asyncio.run(agent.ainvoke({"messages": task}))
    final_output = response["messages"][-1].content
    st.write(final_output)

