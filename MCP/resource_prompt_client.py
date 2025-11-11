import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

HTTP_URL = "http://localhost:8000/mcp"

async def main():
    # Connect over streamable HTTP
    client = MultiServerMCPClient({
        "demo": {"url": HTTP_URL, "transport": "streamable_http"}
    })

    # 1) Get bio from the resource
    blobs = await client.get_resources(server_name="demo",
                                       uris="docs://aboutme")

    bio_text = blobs[0].as_string() if blobs else ""
    print("Bio:", bio_text[:120], "...")

    # 2) Build prompt messages using the bio as context
    messages = await client.get_prompt(server_name="demo",
                                       prompt_name="question",
                                       arguments={
                                           "question":"what subjects does barath teaach?",
                                           "context": bio_text
                                       })

    # 3) Send to LLM
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    resp = await llm.ainvoke(messages)
    print("\nLLM Answer:\n", resp.content)

if __name__ == "__main__":
    asyncio.run(main())