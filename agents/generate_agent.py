import os
import json
import asyncio
from datetime import datetime

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage

from tools.web_search import SimpleSearchFetcher
from rag.rag_store import RagStore

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Initialize a rich Console for status spinners and panels
console = Console()

async def run_generate(topic: str, config_path: str = "config/ollama_config.json"):
    # 1) Load Ollama client
    with console.status("[bold blue]Loading Ollama client…"):
        cfg = json.load(open(config_path, encoding="utf-8"))
        client = OllamaChatCompletionClient(**cfg)

    # 2) RAG context lookup
    with console.status("[bold blue]Performing related RAG lookup…"):
        rag_store = RagStore()
        rag_hits = rag_store.query(topic, top_k=3)
    if not rag_hits:
        console.print("[bold yellow]No prior RAG context found.[/]")
    context_section = "\n\n".join(rag_hits)

    # 3) Web search + fetch
    sf = SimpleSearchFetcher(num_results=3)
    pages = sf.run(topic)  # now also writes into web_content/

    # pages is a dict[url->snippet]
    web_sections = []
    for url, snippet in pages.items():
        web_sections.append(f"### {url}\n\n{snippet}")

    web_findings_md = "\n\n---\n\n".join(web_sections)
    report_md = (
        f"# Report on: {topic}\n\n"
        "## Web Findings\n\n"
        f"{web_findings_md}\n\n"
        "## Analysis and Recommendations\n"
    )

    # 4) Fallback if no content
    if not rag_hits and not pages:
        console.print("[bold yellow]No content available to generate report — asking model directly.[/]")
        fallback_msg = UserMessage(content=topic, source="user")
        with console.status("[bold blue]Generating fallback report…"):
            result = await client.create([fallback_msg])
            summary = getattr(result, "content", None) or str(result)

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(
            c if c.isalnum() or c in ('-', '_') else '_' for c in topic.lower()
        )
        out_dir = "reports"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ts}_{safe}_fallback.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary)
        console.print(Panel(f"Fallback report saved to {out_path}", title="Fallback Report", border_style="green"))
        return

    # 5) Build report markdown
    with console.status("[bold blue]Building report markdown…"):
        report_md = (
            f"# Research Report on {topic}\n\n"
            f"## RAG Context\n\n{context_section}\n\n"
            f"## Web Findings\n\n" + "\n\n---\n\n".join(pages)
        )

    # 6) Summarize via Ollama
    system_msg = SystemMessage(content=(
    "You are a concise research assistant.  "
    "Your first goal is to **directly answer** the user’s **Question**.  "
    "Use the RAG Context or Web Findings **only if they help** illustrate or support your answer.  "
    "If the context is unrelated, ignore it.  "
    "Finally, produce a cohesive Markdown report that includes both the **Answer** and a brief **Summary**."
    ))


    user_msg = UserMessage(content=report_md, source="user")
    with console.status("[bold blue]Summarizing report…"):
        result = await client.create([system_msg, user_msg])  # type: ignore
        summary = getattr(result, "content", None) or str(result)

    # 7) Save markdown
    with console.status("[bold blue]Saving report…"):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(
            c if c.isalnum() or c in ('-', '_') else '_' for c in topic.lower()
        )
        out_dir = "reports"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ts}_{safe}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary)
    console.print(Panel(f"Report saved to {out_path}", title="Report Saved", border_style="green"))

    # 8) Print report summary in Markdown
    with console.status("[bold blue]Rendering report summary…"):
        console.print(Panel(Markdown(summary), title="Report Summary", border_style="green"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/] python generate_agent.py <topic>")
        sys.exit(1)
    topic = " ".join(sys.argv[1:])
    asyncio.run(run_generate(topic))
