import os
import json
import asyncio
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.ollama import OllamaChatCompletionClient

from tools.searxng_search import SearxngSearchTool
from tools.fetch_webpage import FetchWebpageTool
from rag.rag_store import RagStore

async def run_generate(topic: str, config_path="config/ollama_config.json"):
    # Load Ollama client
    cfg = json.load(open(config_path))
    client = OllamaChatCompletionClient(**cfg)

    searcher = SearxngSearchTool()
    fetcher  = FetchWebpageTool()
    rag_store = RagStore()

    # RAG context lookup
    print("Performing related RAG lookup…")
    rag_hits = rag_store.query(topic, top_k=3)
    context = "\n\n".join(rag_hits) if rag_hits else "No prior context found."

    # Fresh web search & fetch
    print(f"Searching web for: {topic}")
    web_hits = searcher.search(topic)
    pages = []
    for hit in web_hits:
        url = hit["link"]
        print(f" Fetching {url}")
        try:
            pages.append(fetcher.fetch(url))
        except Exception as e:
            print(f" Failed to fetch {url}: {e}")

    report_md = (
        f"# Research Report on {topic}\n\n"
        f"## RAG Context\n\n{context}\n\n"
        f"## Web Findings\n\n"
        + "\n\n---\n\n".join(pages)
    )

    # Summarize with Ollama
    agent = AssistantAgent(
        name="Generator",
        llm=client,
        system_prompt="You are a concise research summarizer.",
        conditions=[MaxMessageTermination(1)],
    )
    summary = await agent.run(input=f"Summarize this research into a concise report:\n\n{report_md}")

    # Save markdown
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe = topic.replace(" ", "_").lower()
    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ts}_{safe}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f" Report saved to {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_agent.py <topic>")
        sys.exit(1)
    asyncio.run(run_generate(" ".join(sys.argv[1:])))
