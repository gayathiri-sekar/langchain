import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import api_key

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
question=input("What is the question?")
response = llm.invoke(question)
print(response.content)