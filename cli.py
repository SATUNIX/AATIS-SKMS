# menu.py – CLI hub with RAG Expander

import sys
import asyncio
import signal
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

from agents.generate_agent import run_generate
from agents.generate_agent_large import run_generate as run_generate_large
from agents.qa_agent import run_qa
from agents.expander import expand_once                 # ← NEW
from rag.ingest import refresh_store

console = Console()

MENU_TEXT = (
    "[1] Generate Report\n"
    "[2] Generate Large Report\n"
    "[3] Ask a Question\n"
    "[4] Refresh RAG Reports\n"
    "[5] Expand RAG (auto)\n"
    "[6] Exit"
)


async def handle_choice(choice: str) -> None:
    if choice == "1":
        topic = Prompt.ask("[bold green]Enter topic for report[/]")
        await run_generate(topic)

    elif choice == "2":
        topic = Prompt.ask("[bold green]Enter topic for large report[/]")
        await run_generate_large(topic)

    elif choice == "3":
        question = Prompt.ask("[bold green]Enter your question[/]")
        await run_qa(question)

    elif choice == "4":
        with console.status("[bold blue]Refreshing RAG reports…"):
            refresh_store()
        console.print(
            Panel(
                "[bold green]Refresh complete![/]",
                title="RAG Refresh",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    elif choice == "5":
        taxonomy_path = Prompt.ask(
            "[bold green]Path to taxonomy file[/]",
            default="data/taxonomy.txt",
        )
        top_n = int(
            Prompt.ask("[bold green]How many topics?[/]", default="25")
        )
        num_results = int(
            Prompt.ask("[bold green]Web results per topic?[/]", default="5")
        )

        with console.status("[bold blue]Running expansion…"):
            new_vecs = expand_once(
                taxonomy=Path(taxonomy_path),
                top_n=top_n,
                num_results=num_results,
            )

        console.print(
            Panel(
                f"[bold green]{new_vecs} new vectors added.[/]"
                if new_vecs
                else "[bold yellow]No new vectors added.[/]",
                title="RAG Expansion",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    elif choice == "6":
        console.print(
            Panel(
                "Goodbye!",
                title="Exit",
                border_style="magenta",
                box=box.ROUNDED,
            )
        )
        sys.exit(0)

    else:
        console.print(
            Panel(
                "Invalid choice, please try again.",
                title="Error",
                border_style="red",
                box=box.ROUNDED,
            )
        )


def display_menu() -> None:
    console.print(
        Panel(
            MENU_TEXT,
            title="Main Menu",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def main() -> None:
    def _exit_gracefully(signum, frame):  # noqa: ANN001
        console.print(
            Panel(
                "Goodbye!",
                title="Exit",
                border_style="magenta",
                box=box.ROUNDED,
            )
        )
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit_gracefully)

    while True:
        try:
            display_menu()
            choice = Prompt.ask(
                "Select an option",
                choices=["1", "2", "3", "4", "5", "6"],
            )
            asyncio.run(handle_choice(choice))
        except KeyboardInterrupt:
            _exit_gracefully(None, None)


if __name__ == "__main__":
    main()