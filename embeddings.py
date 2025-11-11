import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import api_key

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
text=input("What is the text?")
response = llm.embed_query(text)
print(response)