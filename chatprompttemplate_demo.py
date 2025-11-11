import os

from click import prompt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import api_key
import streamlit as st
from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

set_debug(True)

llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an agile coach. Answer any questions related to agile process"),
        ("human","{input}")
    ]
)
st.title("Agile Guide")
input=st.text_input("Enter the question:")
chain=prompt_template | llm

if input:
    response = chain.invoke({"input":input})
    st.write(response.content)