# agents/generate_agent.py

import os
import json
import asyncio
from datetime import datetime

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage

from tools.searxng_search import SearxngSearchTool
from tools.fetch_webpage import FetchWebpageTool
from rag.rag_store import RagStore

async def run_generate(topic: str, config_path="config/ollama_config.json"):
    # 1) Load Ollama client
    cfg = json.load(open(config_path, encoding="utf-8"))
    client = OllamaChatCompletionClient(**cfg)

    # 2) RAG context lookup
    rag_store = RagStore()
    print("Performing related RAG lookup…")
    rag_hits = rag_store.query(topic, top_k=3)
    if not rag_hits:
        print("No prior RAG context found.")
    context_section = "\n\n".join(rag_hits)

    # 3) Web search + fetch ++++++++++++++++ YES I KNOW ITS BROKEN RIGHT NOW 
    searcher = SearxngSearchTool()
    fetcher  = FetchWebpageTool()
    print(f"Searching web for: {topic}")
    web_hits = searcher.search(topic)
    if not web_hits:
        print("No search results found.")
    pages = []
    for hit in web_hits:
        url = hit["link"]
        print(f"Fetching {url}")
        try:
            pages.append(fetcher.fetch(url))
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    # 4) If no content, fallback to asking the model directly
    if not rag_hits and not pages:
        print("No content available to generate report — asking model directly.")
        fallback_msg = UserMessage(content=topic, source="user")
        result = await client.create([fallback_msg])
        summary = getattr(result, "content", None) or str(result)

        ts   = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(c if c.isalnum() or c in ('-','_') else '_' for c in topic.lower())
        out_dir = "reports"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ts}_{safe}_fallback.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Fallback report saved to {out_path}")
        return

    # 5) Build report markdown
    report_md = (
        f"# Research Report on {topic}\n\n"
        f"## RAG Context\n\n{context_section}\n\n"
        f"## Web Findings\n\n"
        + "\n\n---\n\n".join(pages)
    )

    # 6) Summarize via Ollama
    system_msg = SystemMessage(content="You are a concise research summarizer.")
    user_msg   = UserMessage(content=report_md, source="user")
    result = await client.create([system_msg, user_msg])  # type: ignore
    # result.content is typical; fallback to str(result)
    summary = getattr(result, "content", None) or str(result)

    # 7) Save markdown
    ts   = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(c if c.isalnum() or c in ('-','_') else '_' for c in topic.lower())
    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ts}_{safe}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_agent.py <topic>")
        sys.exit(1)
    asyncio.run(run_generate(" ".join(sys.argv[1:])))
