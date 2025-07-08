# ============================
# rag/expansion/generator.py (new)
# ============================
"""ExpansionGenerator – discovers topics missing or under‑covered in the RAG.

Run it from CLI:

    python -m rag.expansion.generator  --taxonomy taxonomy.txt  --topN 25

First milestone: prints candidate search strings to stdout.  Later stages
can push them to a crawl queue or agent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.rag_store import RagStore


class ExpansionGenerator:
    def __init__(
        self,
        store: RagStore,
        taxonomy_path: Path,
        gap_threshold: float = 0.25,
        coverage_alpha: float = 10.0,  # docs per day minimal coverage
        top_n_tax: int = 100,
    ) -> None:
        self.store = store
        self.gap_threshold = gap_threshold
        self.coverage_alpha = coverage_alpha
        self.top_n_tax = top_n_tax

        # load taxonomy labels
        self.tax_labels: List[str] = [l.strip() for l in taxonomy_path.read_text().splitlines() if l.strip()]
        self.tax_embeddings = store.embed.encode(self.tax_labels, convert_to_numpy=True, normalize_embeddings=True)

    # ------------------------------------------------------------------ #
    # public entry‑point

    def run(self) -> None:
        missing = self._taxonomy_gap_scan()
        shallow = self._coverage_scan()
        unmet   = self._low_recall_scan()

        suggestions = sorted(missing | shallow | unmet)
        for topic in suggestions[: self.top_n_tax]:
            print(topic)

    # ------------------------------------------------------------------ #
    # scanners

    def _taxonomy_gap_scan(self) -> Set[str]:
        """Return taxonomy labels with cosine < threshold to nearest centroid."""
        missing: Set[str] = set()
        for label, vec in zip(self.tax_labels, self.tax_embeddings):
            D, _ = self.store.router_search(vec[None, :], top_n=1)
            if D[0][0] < self.gap_threshold:
                missing.add(label)
        return missing

    def _coverage_scan(self) -> Set[str]:
        """Return clusters whose doc/day ratio < alpha; emit their top keyword."""
        shallow: Set[str] = set()
        for cid in self.store.list_clusters():
            stats = self.store.cluster_stats(cid)
            ratio = stats["doc_count"] / max(1, stats["age_days"])
            if ratio < self.coverage_alpha:
                shallow.add(f"Deep dive on {stats.get('keyword', 'cluster '+str(cid))}")
        return shallow

    def _low_recall_scan(self) -> Set[str]:
        unmet: Set[str] = set()
        try:
            for q in self.store.get_low_recall_queries(min_recall=0.3):
                unmet.add(q)
        except AttributeError:
            pass  # fallback when store lacks logging
        return unmet


# ---------------------------------------------------------------------- #
# CLI glue


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Generate expansion search topics from RAG gaps.")
    p.add_argument("--taxonomy", type=Path, required=True, help="Text file with one concept per line.")
    p.add_argument("--topN", type=int, default=50, help="Max number of topics to print.")
    p.add_argument("--store", type=Path, default=Path("rag/index.faiss"), help="Path to main FAISS index.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])
    store = RagStore(index_path=str(args.store))
    gen = ExpansionGenerator(store, args.taxonomy, top_n_tax=args.topN)
    gen.run()


if __name__ == "__main__":
    main()
