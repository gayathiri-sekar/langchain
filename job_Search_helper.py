import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
# Loads Text files
from langchain_community.document_loaders import TextLoader
# Split document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Vector Store
from langchain_chroma import Chroma

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Load up the document
document = TextLoader("job_listings.txt").load()
# Split the document into data chunks
# Text splitter class - chunk size & chunk overlap - to maintain context between chunks
text_splitter= RecursiveCharacterTextSplitter(chunk_size=200,
                                              chunk_overlap=10)
# Split the docs
chunks=text_splitter.split_documents(document)
# chunks + llm used for embeddings to vector store
db=Chroma.from_documents(chunks,llm)

# Take the query from user
text = input("Enter the query")
# Get the embedding from llm
embedding_vector=llm.embed_query(text)
# Pass the embedding to vector to do similarity search
docs = db.similarity_search_by_vector(embedding_vector)

print(docs[0].page_content)