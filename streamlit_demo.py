import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import api_key
import streamlit as st
from langchain_core.globals import set_debug

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

set_debug(True)

st.title("OpenAI Chat")
llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
question=st.text_input("What is the question?")
if question:
    response = llm.invoke(question)
    st.write(response.content)