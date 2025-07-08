# ============================
# agents/expander.py  (new location – CLI-visible agent)
# ============================
"""RAG Expander – non‑interactive background process that augments the store.

Placed in *agents/* so it shows up in your CLI’s agent picker next to
*generate.py*.  It exposes both a synchronous API (`expand_once`) and a
`main()` CLI entry‑point, but **no LLM calls** – pure automation.

Typical usage:

```bash
# fire‑and‑forget expansion, appending logs
ohup python -m agents.expander \
      --taxonomy data/taxonomy.txt \
      --topN 25 --num-results 5 \
      >> logs/expander.log 2>&1 &
```
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel

from rag.rag_store import RagStore
from rag.expansion.generator import ExpansionGenerator
from tools.web_search import SimpleSearchFetcher

console = Console()

# ---------------------------------------------------------------------- #
# PUBLIC PROGRAMMATIC API


def expand_once(
    taxonomy: Path | str,
    store_path: Path | str = "rag/index.faiss",
    top_n: int = 25,
    num_results: int = 5,
) -> int:
    """Run one expansion cycle and return the number of new vectors added."""

    store = RagStore(index_path=str(store_path))
    gen = ExpansionGenerator(store, taxonomy_path=Path(taxonomy))
    topics = gen.suggestions(top_n=top_n)

    if not topics:
        console.print("[yellow]No expansion topics identified; store appears up‑to‑date.")
        return 0

    console.print(Panel("
".join(topics), title="Expansion Topics", border_style="cyan"))

    total_new = 0
    fetcher = SimpleSearchFetcher(num_results=num_results)

    for topic in topics:
        pages = fetcher.run(topic)
        if not pages:
            console.print(f"[yellow]No search results for '{topic}'.")
            continue

        docs: List[str] = [f"## Source: {url}

{snippet}" for url, snippet in pages.items()]
        pre = store.ntotal()
        store.add_documents(docs)
        added = store.ntotal() - pre
        total_new += added
        console.print(f"[green]Added {added} docs for '{topic}'.")

    if total_new:
        store.save()
        console.print(f"[bold green]Expansion complete → {total_new} new vectors.[/]")
    else:
        console.print("[bold yellow]Expansion yielded no new vectors.")

    return total_new


# ---------------------------------------------------------------------- #
# ASYNC WRAPPER (keeps signature parity with other agents)


async def run_async(args) -> None:
    await asyncio.to_thread(
        expand_once,
        taxonomy=args.taxonomy,
        store_path=args.store,
        top_n=args.topN,
        num_results=args.num_results,
    )


# ---------------------------------------------------------------------- #
# CLI ENTRY‑POINT


def _parse(argv):
    p = argparse.ArgumentParser(description="Automated RAG expansion agent (no LLM).")
    p.add_argument("--taxonomy", type=Path, required=True, help="File with one concept per line.")
    p.add_argument("--topN", type=int, default=25, help="Number of topics to expand.")
    p.add_argument("--num-results", type=int, default=5, help="Web search results per topic.")
    p.add_argument("--store", type=Path, default=Path("rag/index.faiss"))
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()