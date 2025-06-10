import os
import json
import asyncio
from datetime import datetime

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage

from tools.searxng_search import SearxngSearchTool
from tools.fetch_webpage import FetchWebpageTool
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
    searcher = SearxngSearchTool()
    fetcher = FetchWebpageTool()
    with console.status(f"[bold blue]Searching web for: {topic}"):
        web_hits = searcher.search(topic)
    if not web_hits:
        console.print("[bold yellow]No search results found.[/]")
    pages = []
    for hit in web_hits:
        url = hit.get("link")
        with console.status(f"[bold blue]Fetching {url}"):
            try:
                pages.append(fetcher.fetch(url))
            except Exception as e:
                console.print(f"[bold red]Failed to fetch {url}:[/] {e}")

    # 4) Fallback if no content
    if not rag_hits and not pages:
        console.print("[bold yellow]No content available — asking model directly.[/]")
        fallback_msg = UserMessage(content=topic, source="user")
        with console.status("[bold blue]Generating fallback summary…"):
            result = await client.create([fallback_msg])
            summary = getattr(result, "content", None) or str(result)
    else:
        # 5) Build initial report markdown
        with console.status("[bold blue]Building initial report markdown…"):
            report_md = (
                f"# Research Report on {topic}\n\n"
                f"## RAG Context\n\n{context_section}\n\n"
                f"## Web Findings\n\n" + "\n\n---\n\n".join(pages)
            )
        # 6) Summarize via Ollama
        system_msg = SystemMessage(content=(
            "You are a concise research assistant. First, answer the user’s Question directly. "
            "Then integrate any relevant context or findings as evidence. Finally, provide a brief SUMMARY."
        ))
        user_msg = UserMessage(content=report_md, source="user")
        with console.status("[bold blue]Summarizing initial report…"):
            result = await client.create([system_msg, user_msg])
            summary = getattr(result, "content", None) or str(result)

    # 7) Generate follow-up questions
    with console.status("[bold blue]Generating follow-up questions…"):
        q_system = SystemMessage(content=(
            "You are an inquisitive assistant. Based on the following summary, "
            "generate three insightful follow-up questions that deepen understanding."
        ))
        q_user = UserMessage(content=summary, source="user")
        q_result = await client.create([q_system, q_user])
        questions_raw = getattr(q_result, "content", None) or str(q_result)
    # Parse questions into list
    questions = [q.strip() for q in questions_raw.splitlines() if q.strip()]

    # 8) Answer each follow-up question separately
    answers = []
    for idx, question in enumerate(questions, start=1):
        with console.status(f"[bold blue]Answering question {idx}…"):
            a_system = SystemMessage(content=(
                "You are a detailed assistant. Answer the following question thoroughly, "
                "using any relevant context."
            ))
            a_user = UserMessage(content=question, source="user")
            a_result = await client.create([a_system, a_user])
            answer = getattr(a_result, "content", None) or str(a_result)
            answers.append((question, answer))

    # 9) Build and generate final comprehensive report
    with console.status("[bold blue]Compiling final report…"):
        # Prepare final prompt
        final_content = f"# Final Research Report on {topic}\n\n"
        final_content += f"## Initial Summary\n{summary}\n\n"
        final_content += "## Follow-Up Q&A\n"
        for q, a in answers:
            final_content += f"### {q}\n{a}\n\n"
        final_system = SystemMessage(content=(
            "You are a comprehensive research assistant. "
            "Weave the above summary and Q&A into a cohesive, detailed Markdown report."
        ))
        final_user = UserMessage(content=final_content, source="user")
        final_result = await client.create([final_system, final_user])
        final_report = getattr(final_result, "content", None) or str(final_result)

    # 10) Save final report
    with console.status("[bold blue]Saving final report…"):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in topic.lower())
        out_dir = "reports"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ts}_{safe}_large.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_report)
    console.print(Panel(f"Final large report saved to {out_path}", title="Report Saved", border_style="green"))

    # 11) Display final report
    with console.status("[bold blue]Rendering final report…"):
        console.print(Panel(Markdown(final_report), title="Detailed Report", border_style="green"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/] python generate_agent_large.py <topic>")
        sys.exit(1)
    topic = " ".join(sys.argv[1:])
    asyncio.run(run_generate(topic))
