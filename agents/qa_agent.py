import json
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.ollama import OllamaChatCompletionClient

from rag.rag_store import RagStore

async def run_qa(question: str, config_path="config/ollama_config.json"):
    # Load Ollama client
    cfg = json.load(open(config_path, encoding="utf-8"))
    client = OllamaChatCompletionClient(**cfg)

    # RAG retrieve
    rag_store = RagStore()
    docs = rag_store.query(question, top_k=5)
    context = "\n\n".join(docs) if docs else "No relevant context found."

    # Ask via Ollama
    agent = AssistantAgent(
        name="QA",
        llm=client,
        system_prompt="You are a knowledgeable assistant. Use the provided context first, then your own knowledge.",
        conditions=[MaxMessageTermination(1)],
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    answer = await agent.run(input=prompt)
    print(answer)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qa_agent.py <question>")
        sys.exit(1)
    asyncio.run(run_qa(" ".join(sys.argv[1:])))
