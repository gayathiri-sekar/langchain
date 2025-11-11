# ------------------------------------------------------------
#  IMPORTS AND SETUP
# ------------------------------------------------------------
import os
from dotenv import load_dotenv  # For loading environment variables from a .env file

# LangChain modular packages (as per v1.x architecture)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI API wrappers
from langchain_community.document_loaders import TextLoader  # Load plain text files as documents
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split long text into overlapping chunks
from langchain_chroma import Chroma  # Vector store backed by ChromaDB

# LangChain Core building blocks (used to define LLM pipelines)
from langchain_core.prompts import ChatPromptTemplate  # Structured prompt templates for chat models
from langchain_core.output_parsers import StrOutputParser  # Converts LLM outputs into clean text strings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # LCEL “pipeline” utilities

# ------------------------------------------------------------
#  LOAD ENVIRONMENT VARIABLES (API KEY, ETC.)
# ------------------------------------------------------------
load_dotenv()  # Load variables from .env file into environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Get your OpenAI key safely from environment

# ------------------------------------------------------------
#  INITIALIZE LLM AND EMBEDDING MODELS
# ------------------------------------------------------------
# Create an embedding model to convert text into numerical vectors for semantic search
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Create a chat-based LLM instance (GPT-4o here, can be replaced with other supported models)
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# ------------------------------------------------------------
#  LOAD AND PREPARE DOCUMENTS
# ------------------------------------------------------------
# Load a text document (each line or section is treated as a Document object)
docs = TextLoader("rag_files/product-data.txt").load()

# Split the loaded document into smaller overlapping chunks
# - chunk_size=1000 defines how many characters per chunk
# - chunk_overlap=200 ensures some overlap to maintain context between chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)  # Perform the actual split

# ------------------------------------------------------------
#  CREATE VECTOR STORE AND RETRIEVER
# ------------------------------------------------------------
# Store document chunks as embeddings in a Chroma vector store (in-memory by default)
vector_store = Chroma.from_documents(chunks, embeddings)

# Convert the vector store into a retriever interface (for similarity search)
retriever = vector_store.as_retriever()

# ------------------------------------------------------------
#  DEFINE PROMPT TEMPLATE FOR THE LLM
# ------------------------------------------------------------
# Create a system + human prompt template that:
# - provides context (retrieved chunks)
# - asks the user’s question
# - instructs the model to answer concisely
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for answering questions. "
            "Use the provided context to respond. If the answer isn't clear, acknowledge that you don't know. "
            "Limit your response to three concise sentences.\n\n{context}"
        ),
        ("human", "{input}"),  # User question will be inserted here dynamically
    ]
)

# ------------------------------------------------------------
#  HELPER FUNCTION TO FORMAT RETRIEVED DOCUMENTS
# ------------------------------------------------------------
# Converts a list of Document objects into a single string (joined by double newlines)
def join_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ------------------------------------------------------------
#  BUILD THE RETRIEVAL-AUGMENTED GENERATION (RAG) CHAIN
# ------------------------------------------------------------
# The chain flow works as follows:
#   1. Take user input → Retrieve relevant chunks from vector DB
#   2. Combine retrieved chunks into a single context string
#   3. Pass context + input into the chat prompt
#   4. Send prompt to the LLM
#   5. Parse the LLM output into a clean string
rag_chain = (
    {
        # “context” comes from the retriever (fetch docs) → format with join_docs
        "context": retriever | RunnableLambda(join_docs),
        # “input” simply passes through the user question unchanged
        "input": RunnablePassthrough(),
    }
    # Combine the inputs into the ChatPromptTemplate
    | prompt
    # Feed the filled prompt into the ChatOpenAI model
    | llm
    # Parse the model’s response into plain text
    | StrOutputParser()
)

# ------------------------------------------------------------
#  SIMPLE CHAT LOOP (ONE-SHOT INTERACTION)
# ------------------------------------------------------------
print("Chat with Document")  # Prompt header
question = input("Your Question: ")  # Get user input from console

# Only run if user provided a question
if question:
    # Invoke the RAG chain — this automatically performs retrieval + generation
    answer = rag_chain.invoke(question)

    # Print the LLM’s concise, context-aware answer
    print(answer)
