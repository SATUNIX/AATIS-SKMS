import json
import asyncio
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from autogen_core.models import SystemMessage, UserMessage
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

    # 1. Run the agent
    with console.status("[bold blue]Running agent..."):
        result = await agent.run(task=prompt)

    # 2. Extract the last message (raw agent output)
    if not (hasattr(result, "messages") and result.messages):
        console.print("[bold red]ERROR[/] Agent returned no usable output.")
        console.print(Panel(str(result), title="Raw Agent Output", border_style="red"))
        return

    # NB: use .to_text() instead of .content
    last_msg = result.messages[-1]
    raw_output = last_msg.to_text().strip()

    # 3. Display the raw agent response as Markdown
    console.print(Panel(Markdown(raw_output), title="Agent Response", border_style="green"))

    # 4. Summarize the raw output
    with console.status("[bold blue]Summarizing output..."):
        system_msg = SystemMessage(
            content=(
                "You are a concise research summarizer. Summarize this information "
                "and keep all key details, ensuring your output is a full markdown "
                "report written in the style of a message."
            )
        )
        user_msg = UserMessage(content=raw_output, source="user")
        try:
            summary_result = await client.create([system_msg, user_msg])  # type: ignore
            # Depending on your client you may still need to pull out .chat_message or .choices
            summary = getattr(summary_result, "content", None) or str(summary_result)
            summary = summary.strip()
        except Exception as e:
            console.print(f"[yellow]WARN[/] Summarization failed: {e}")
            return

    # 5. Display the summarized Markdown
    if summary:
        console.print(Panel(Markdown(summary), title="QA Answer", border_style="green"))
    else:
        console.print("[bold red]ERROR[/] Summarization returned no content.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py ask <question>", file=sys.stderr)
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    try:
        asyncio.run(run_qa(q))
    except Exception as e:
        print(f"[FATAL] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
