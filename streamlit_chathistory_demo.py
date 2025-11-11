import os

from click import prompt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import api_key
import streamlit as st
from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

set_debug(True)

llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an agile coach. Answer any questions related to agile process"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
)
chain=prompt_template | llm
history_for_chain=StreamlitChatMessageHistory()
chain_with_history=RunnableWithMessageHistory(
    chain,
    lambda session_id:history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Input in browser
st.title("Agile Guide")
input=st.text_input("Enter the question:")

if input:
    response = chain_with_history.invoke({"input":input}, {"configurable":{"session_id":"abc123"}})
    st.write(response.content)
    # Print history
    st.write("HISTORY")
    st.write(history_for_chain)