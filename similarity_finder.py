import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import api_key
import numpy as np

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
text1=input("What is the text1?")
text2=input("What is the text2?")
response1 = llm.embed_query(text1)
response2 = llm.embed_query(text2)

similarity_score = np.dot(response1, response2)
print(similarity_score*100,'%')