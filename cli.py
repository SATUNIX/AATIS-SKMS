import sys
import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

from agents.generate_agent import run_generate
from agents.generate_agent_large import run_generate as run_generate_large
from agents.qa_agent import run_qa
from rag.ingest_reports import refresh_reports

console = Console()

MENU_TEXT = (
    "[1] Generate Report\n"
    "[2] Generate Large Report\n"
    "[3] Ask a Question\n"
    "[4] Refresh RAG Reports\n"
    "[5] Exit"
)

async def handle_choice(choice: str):
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
            refresh_reports()
        console.print(Panel("[bold green]Refresh complete![/]", title="RAG Refresh", border_style="green", box=box.ROUNDED))
    elif choice == "5":
        console.print(Panel("Goodbye!", title="Exit", border_style="magenta", box=box.ROUNDED))
        sys.exit(0)
    else:
        console.print(Panel("Invalid choice, please try again.", title="Error", border_style="red", box=box.ROUNDED))


def display_menu():
    console.clear()
    console.print(Panel(MENU_TEXT, title="Main Menu", border_style="cyan", box=box.ROUNDED, padding=(1, 2)))


def main():
    while True:
        display_menu()
        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"])
        asyncio.run(handle_choice(choice))


if __name__ == "__main__":
    main()
