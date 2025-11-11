import os

from click import prompt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import api_key
import streamlit as st
from langchain_core.globals import set_debug
from langchain_core.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

set_debug(True)

llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
prompt_template = PromptTemplate(
    input_variables=["city", "month", "budget","language"],
    template="""Welcome to the {city} travel guide!
    If you're visiting in {month}, here's what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.
    Enjoy your trip!
    """
)
st.title("Travel Guide")
city=st.text_input("Enter the city:")
month=st.text_input("Enter the month of travel:")
language = st.text_input("Enter the language:")
budget = st.selectbox("Travel Budget:",["Low","Medium","High"])

if city and month and budget and language:
    response = llm.invoke(prompt_template.format(city=city,month=month,budget=budget,language=language))
    st.write(response.content)