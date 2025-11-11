# ------------------------------------------------------------
#  IMPORTS AND SETUP
# ------------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv  # Loads environment variables from .env file

# --- LangChain modular imports (no 'langchain.chains' used) ---
# OpenAI wrappers for embeddings and LLM (GPT models)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Loaders, text splitters, and vector store connectors
# Loads files into Document objects
from langchain_community.document_loaders import (
    TextLoader,          # text / md files
    CSVLoader,           # csv files
    PyPDFLoader,         # pdf files
    Docx2txtLoader       # docx files
)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text into overlapping chunks
from langchain_chroma import Chroma  # Vector store backend for storing embeddings

# Core building blocks for LangChain Expression Language (LCEL)
from langchain_core.prompts import (
    ChatPromptTemplate,  # Used to define structured chat-style prompts
    PromptTemplate,      # Used for simple text-based prompts
    MessagesPlaceholder, # Placeholder to inject chat history dynamically
)
from langchain_core.output_parsers import StrOutputParser  # Converts LLM outputs into plain strings
from langchain_core.runnables import (
    RunnablePassthrough,  # For passing inputs unchanged through a chain
    RunnableLambda,        # For applying inline Python logic within a chain
)
from langchain_core.runnables.history import RunnableWithMessageHistory  # Adds persistent chat history
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # Chat history store for Streamlit apps

# ------------------------------------------------------------
#  ENVIRONMENT AND MODEL INITIALIZATION
# ------------------------------------------------------------
# Load environment variables (e.g., OPENAI_API_KEY) from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create an embedding model instance for converting text to numerical vectors
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize the LLM (using GPT-4o here) for both query rewriting and answering
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# ------------------------------------------------------------
#  LOAD AND PREPARE DOCUMENTS
# ------------------------------------------------------------
# Load ALL supported documents inside rag_files/ (recursively).
# Supported extensions: .txt, .md/.markdown, .csv, .pdf
def load_all_docs(folder: str):
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            try:
                if ext in {".txt", ".md", ".markdown"}:
                    docs.extend(TextLoader(path, encoding="utf-8").load())
                elif ext == ".csv":
                    # CSVLoader reads entire file; treat each row as a doc
                    docs.extend(CSVLoader(file_path=path, encoding="utf-8").load())
                elif ext == ".pdf":
                    docs.extend(PyPDFLoader(path).load())
                elif ext == ".docx":
                    # Uses docx2txt under the hood to extract text from Word documents
                    docs.extend(Docx2txtLoader(path).load())
                else:
                    # Skip unsupported types silently (or print if you prefer)
                    # print(f"Skipping unsupported file type: {path}")
                    pass
            except Exception as e:
                print(f"Skipping {path} due to error: {e}")

    return docs

docs = load_all_docs("rag_files")

# Split the loaded document into overlapping chunks so context is preserved between splits
# - chunk_size=1000 → number of characters per chunk
# - chunk_overlap=200 → ensures smooth continuity between adjacent chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# ------------------------------------------------------------
#  CREATE VECTOR STORE AND RETRIEVER
# ------------------------------------------------------------
# Create a Chroma vector store from the chunk embeddings (stored locally/in-memory)
vector_store = Chroma.from_documents(chunks, embeddings)

# Convert the Chroma store into a retriever interface to perform similarity searches
retriever = vector_store.as_retriever()

# ------------------------------------------------------------
#  PROMPTS DEFINITIONS
# ------------------------------------------------------------

# --- Rewriting Prompt ---
# This LLM prompt takes the chat history + current user query
# and rewrites the query into a standalone question.
# Useful for follow-ups like: "What about pricing?" → becomes → "What is the pricing for Product X?"
rewrite_prompt = PromptTemplate.from_template(
    "Rewrite the user's query into a clear standalone question using the chat history.\n"
    "Chat history:\n{chat_history}\n\n"
    "User query: {input}\n"
    "Standalone question:"
)

# --- Answering Prompt ---
# This is the main QA prompt. It injects:
#   - retrieved 'context' (text chunks relevant to the query)
#   - chat_history (for continuity)
#   - user 'input' (the actual question)
# The system message instructs the LLM to give concise, factual answers.
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for answering questions. Use the provided context "
            "and chat history to respond. If the answer isn't clear, acknowledge "
            "that you don't know. Limit your response to three concise sentences.\n\n"
            "Context:\n{context}"
        ),
        # Insert the dynamic chat history placeholder between system and user
        MessagesPlaceholder(variable_name="chat_history"),
        # The user's latest question
        ("human", "{input}"),
    ]
)

# ------------------------------------------------------------
#  HELPER FUNCTIONS
# ------------------------------------------------------------

# Helper: join a list of Document objects into one large text string (for prompt stuffing)
def join_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Helper: extract chat history and input from a dictionary input structure
get_input = RunnableLambda(lambda x: x["input"])
get_history = RunnableLambda(lambda x: x.get("chat_history", []))

# ------------------------------------------------------------
#  BUILD PIPELINE USING LCEL (LangChain Expression Language)
# ------------------------------------------------------------

# Rewrite Chain:
# This stage rewrites the user query into a standalone question using chat history.
#   Input: {"input": question, "chat_history": serialized_history}
#   Output: rewritten question string
rewrite_chain = (
    {"input": get_input, "chat_history": get_history}  # feed both input & history
    | rewrite_prompt  # apply rewriting prompt
    | llm             # call the LLM to rewrite
    | StrOutputParser()  # extract plain text
)

# Context Retrieval Chain:
# Takes the rewritten query, retrieves top-matching chunks, and merges them into one context string.
#   Input: {"input": question, "chat_history": history}
#   Output: large text block of relevant context
context_chain = rewrite_chain | retriever | RunnableLambda(join_docs)

# Question-Answering Chain ("stuff documents" behavior):
# Combines context + input + chat_history, formats with qa_prompt, and runs the LLM.
#   Equivalent to `create_stuff_documents_chain` from old API.
qa_chain = (
    {
        "context": context_chain,    # retrieved and joined document context
        "input": get_input,          # original user query
        "chat_history": get_history, # serialized prior turns
    }
    | qa_prompt   # format into final chat-style prompt
    | llm         # generate model output
    | StrOutputParser()  # parse LLM output to string
)

# Wrap in Retrieval Chain Output:
# Wraps QA output in a dict {"answer": text} to match original structure expected in Streamlit UI.
rag_chain = qa_chain | RunnableLambda(lambda text: {"answer": text})

# ------------------------------------------------------------
#  STREAMLIT APP + MEMORY INTEGRATION
# ------------------------------------------------------------

# Display title in Streamlit UI
st.write("### 💬 Chat with Document")

# Initialize Streamlit-specific message history handler
# This stores conversation turns (user ↔ assistant) persistently during the session.
history_for_chain = StreamlitChatMessageHistory()

# Wrap our LCEL chain with persistent message memory
# RunnableWithMessageHistory automatically passes chat history between turns
chain_with_history = RunnableWithMessageHistory(
    rag_chain,                                # our RAG pipeline
    lambda session_id: history_for_chain,     # function returning message store
    input_messages_key="input",               # key to extract user messages from input
    history_messages_key="chat_history",      # key where chat history will be injected
)

# Streamlit input field for user questions
question = st.text_input("Ask a question about the document:")

# If user submits a question
if question:
    # Run the chain, providing session_id (to persist history uniquely per user/session)
    response = chain_with_history.invoke(
        {"input": question},
        {"configurable": {"session_id": "gaya123"}}
    )

    # Display the final answer in Streamlit
    st.write("**Answer:**", response["answer"])


# Summary of Pipeline Flow
# ------------------------
# 1. User enters a question →
# The input goes into chain_with_history, which maintains conversation context.
# 2. Query Rewriting →
# The rewrite_chain uses the LLM to rephrase follow-ups into self-contained questions.
# 3. Retrieval →
# The rewritten query is passed to retriever, which performs a semantic search in Chroma.
# 4. Document Stuffing →
# Retrieved chunks are joined and injected into a structured qa_prompt.
# 5. LLM Answering →
# The ChatOpenAI model generates a concise, context-aware response.
# 6. Display + Memory →
# RunnableWithMessageHistory automatically logs user–assistant messages in StreamlitChatMessageHistory, enabling coherent multi-turn chat.
# ------------------------------------------------------------------------