import json
import asyncio
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from rag.rag_store import RagStore

# Single global console instance
console = Console()

async def run_qa(question: str, config_path="config/ollama_config.json"):
    # 1. Load Ollama client config
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]ERROR[/] Config file not found: {config_path}")
        return
    except json.JSONDecodeError as e:
        console.print(f"[bold red]ERROR[/] Invalid JSON in {config_path}: {e}")
        return

    # 2. Initialize Ollama client
    try:
        client = OllamaChatCompletionClient(**cfg)
    except TypeError as e:
        console.print(f"[bold red]ERROR[/] Bad config keys for OllamaChatCompletionClient: {e}")
        return
    except Exception as e:
        console.print(f"[bold red]ERROR[/] Failed to initialize Ollama client: {e}")
        return

    # 3. Fetch RAG context
    try:
        rag_store = RagStore()
        docs = rag_store.query(question, top_k=5)
        context = "\n\n".join(docs) if docs else "No relevant context found."
    except Exception as e:
        console.print(f"[yellow]WARN[/] RAG retrieval failed, proceeding without context: {e}")
        context = "No relevant context found."

    # 4. Set up the AssistantAgent
    agent = AssistantAgent(
        name="QA",
        model_client=client,
        system_message=(
            "You are a knowledgeable assistant. "
            "Use the provided context first, then your own knowledge.  \n\n"
            "**Output your answer in Markdown format.**"
        ),
    )

    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # 5. Run the agent
    try:
        result = await agent.run(task=prompt)
    except Exception as e:
        console.print(f"[bold red]ERROR[/] Agent run failed: {e}")
        return

    # 6. Extract and print output cleanly
    if hasattr(result, "output") and isinstance(result.output, str):
        answer = result.output.strip()
        console.print(Panel(Markdown(answer), title="🧠 QA Answer", border_style="green"))
    else:
        console.print("[bold red]ERROR[/] Agent returned no usable output.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qa_agent.py <question>", file=sys.stderr)
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    try:
        asyncio.run(run_qa(q))
    except Exception as e:
        print(f"[FATAL] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
