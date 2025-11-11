import os

from click import prompt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import api_key
import streamlit as st
from langchain_core.globals import set_debug
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

set_debug(True)

llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
        You need to craft an impactful title for a speech
        on the following topic: {topic}
        Answer exactly with one title.
        """
)
speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""You need to write a powerful speech of 350 words
        for the following title: {title}
        """
)

first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1])
second_chain = speech_prompt | llm
final_chain = first_chain | second_chain

st.title("Speech Generator")
topic=st.text_input("Enter the topic:")

if topic:
    response = final_chain.invoke({
        "topic": topic
    })
    st.write(response.content)