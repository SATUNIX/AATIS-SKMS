# main.py   — top-level CLI router
import sys
import asyncio

from agents.generate_agent import run_generate
from agents.generate_agent_large import run_generate as run_generate_large
from agents.qa_agent import run_qa
from agents.expander import expand_once          # ← NEW
from rag.ingest import refresh_store


def main() -> None:
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    # ------------------------------------------------------------------ #
    # content-generation agents
    # ------------------------------------------------------------------ #
    if cmd == "generate":
        _require_args(3, "topic")
        topic = " ".join(sys.argv[2:])
        asyncio.run(run_generate(topic))

    elif cmd == "generate-large":
        _require_args(3, "topic")
        topic = " ".join(sys.argv[2:])
        asyncio.run(run_generate_large(topic))

    elif cmd == "ask":
        _require_args(3, "question")
        question = " ".join(sys.argv[2:])
        asyncio.run(run_qa(question))

    # ------------------------------------------------------------------ #
    # maintenance / expansion
    # ------------------------------------------------------------------ #
    elif cmd == "refresh":
        refresh_store()

    elif cmd == "expand":
        # Syntax:  python main.py expand [taxonomy_path] [topN] [num_results]
        taxonomy   = sys.argv[2] if len(sys.argv) >= 3 else "data/taxonomy.txt"
        top_n      = int(sys.argv[3]) if len(sys.argv) >= 4 else 25
        num_results = int(sys.argv[4]) if len(sys.argv) >= 5 else 5
        expand_once(taxonomy=taxonomy, top_n=top_n, num_results=num_results)

    else:
        print(f"Unknown command: {cmd}")
        _usage()
        sys.exit(1)


# ---------------------------------------------------------------------- #
# utilities
# ---------------------------------------------------------------------- #
def _usage() -> None:
    print(
        "Usage:\n"
        "  python main.py generate <topic>\n"
        "  python main.py generate-large <topic>\n"
        "  python main.py ask <question>\n"
        "  python main.py refresh\n"
        "  python main.py expand [taxonomy_path] [topN] [num_results]\n"
        "      – defaults: taxonomy_path=data/taxonomy.txt, topN=25, num_results=5"
    )


def _require_args(threshold: int, name: str) -> None:
    if len(sys.argv) < threshold:
        print(f"Error: Missing {name}.")
        _usage()
        sys.exit(1)


if __name__ == "__main__":
    main()