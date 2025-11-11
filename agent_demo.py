import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import langchainhub as hub
import streamlit as st
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic import hub

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

prompt = hub.pull("hwchase17/react")
tools = load_tools(["wikipedia","ddg-search"])

agent= create_react_agent(llm,tools,prompt)
agent_executor=AgentExecutor(agent= agent,tools=tools,verbose=True)

st.title("AI Agent")
task=st.text_input("Assign me a task")

if task:
    response = agent_executor.invoke({"input":task})
    st.write(response["output"])