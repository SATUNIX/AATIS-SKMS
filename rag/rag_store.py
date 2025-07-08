# ============================
# rag/rag_store.py (updated)
# ============================
from __future__ import annotations

import hashlib
import math
import os
import pickle
from pathlib import Path
from typing import List, Sequence

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer


class RagStore:
    """Hierarchy‑aware FAISS store with automatic index selection.

    For <1 M vectors → HNSW32,Flat (exact search via graph).
    For 1–30 M      → IVF4096_HNSW32,PQ96x8fsr,RefineFlat  (≈log N).
    For >30 M       → IVF16384,PQ96x8fsr  (on‑disk friendly).

    The class transparently rebuilds the index when vector count crosses
    a tier boundary, so callers never need to care which backend is in
    use.
    """

    # ---------- thresholds & factories --------------------------------- #
    _TIER_CFG = [
        (1_000_000, "HNSW32,Flat"),
        (30_000_000, "IVF4096_HNSW32,PQ96x8fsr,RefineFlat"),
        (math.inf, "IVF16384,PQ96x8fsr"),
    ]

    def __init__(
        self,
        index_path: str | Path = "rag/index.faiss",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.embed = SentenceTransformer(embed_model)
        self.dim = self.embed.get_sentence_embedding_dimension()

        self._load_or_init()

        # SHA‑256 digest → doc index, used for dedup and incremental ingest
        self._digest_to_id: dict[str, int] = {
            hashlib.sha256(d.encode("utf-8")).hexdigest(): i
            for i, d in enumerate(self.docs)
        }

    # ------------------------------------------------------------------ #
    # public API

    def add_documents(self, docs: Sequence[str]) -> None:
        """Embed and add *unique* document strings to the index."""
        unique_pairs = [
            (doc, self._hash(doc)) for doc in docs if self._hash(doc) not in self._digest_to_id
        ]
        if not unique_pairs:
            return

        texts, digests = zip(*unique_pairs)
        embs = self.embed.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)

        # Rebuild index if we cross a tier boundary
        will_total = self.index.ntotal + embs.shape[0]
        if self._requires_rebuild(will_total):
            self._rebuild_index(new_vecs=embs, new_texts=list(texts), new_digests=list(digests))
            return

        # Normal incremental add
        start_id = len(self.docs)
        ids = (faiss.IndexIDMap2.numpy_int64_to_idx64(range(start_id, start_id + len(texts))))
        self.index.add_with_ids(embs, ids)  # type: ignore[arg-type]
        self.docs.extend(texts)
        self._digest_to_id.update(dict(zip(digests, ids)))

    def query(self, text: str, top_k: int = 5, mode: str = "auto") -> List[str]:
        """Retrieve *top_k* nearest docs.  *mode* ∈ {auto|breadth|depth}."""
        if self.index.ntotal == 0:
            return []

        q_emb = self.embed.encode([text], convert_to_numpy=True, normalize_embeddings=True)

        # Breadth searches look into more list buckets; depth searches widen nprobe.
        if hasattr(self.index, "nprobe"):
            if mode == "breadth":
                self.index.nprobe = min(64, getattr(self.index, "nlist", 1))
            elif mode == "depth":
                self.index.nprobe = 1
            else:  # auto
                self.index.nprobe = 16

        dists, idxs = self.index.search(q_emb, top_k)  # type: ignore[misc]
        return [self.docs[i] for i in idxs[0] if 0 <= i < len(self.docs)]

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))  # type: ignore[arg-type]
        with open(self.index_path.with_suffix(".meta"), "wb") as f:
            pickle.dump(self.docs, f)

    def ntotal(self) -> int:  # convenience
        return int(self.index.ntotal)

    def is_duplicate(self, doc: str) -> bool:
        return self._hash(doc) in self._digest_to_id

    # ------------------------------------------------------------------ #
    # PRIVATE IMPLEMENTATION

    def _load_or_init(self) -> None:
        meta_path = self.index_path.with_suffix(".meta")
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))  # type: ignore[arg-type]
            with open(meta_path, "rb") as f:
                self.docs: list[str] = pickle.load(f)
        else:
            self.index = self._build_index_factory(0)
            self.docs = []

    # ---------- index‑construction helpers ----------------------------- #
    def _build_index_factory(self, n_vectors_estimate: int):
        factory = self._choose_factory(n_vectors_estimate)
        index = faiss.index_factory(self.dim, factory, faiss.METRIC_INNER_PRODUCT)
        # IVF indexes need training – prepare an empty coarse quantiser now
        if index.is_trained is False:
            # train with random Gaussian so the index is at least initialised;
            # it will be retrained on first real addition.
            dummy = faiss.random_distrib(self.dim, 2048)
            faiss.normalize_L2(dummy)
            index.train(dummy)
        return faiss.IndexIDMap2(index)

    def _choose_factory(self, n_vectors: int) -> str:
        for upper_bound, factory in self._TIER_CFG:
            if n_vectors < upper_bound:
                return factory
        # unreachable
        return "HNSW32,Flat"

    def _requires_rebuild(self, new_total: int) -> bool:
        current_factory = self._extract_factory_string(self.index)
        desired_factory = self._choose_factory(new_total)
        return current_factory != desired_factory

    def _rebuild_index(
        self,
        new_vecs,
        new_texts: list[str],
        new_digests: list[str],
    ) -> None:
        """Full rebuild into a larger‑capacity index tier."""
        print("[RagStore] Rebuilding FAISS index for new scale – please wait…")
        all_vecs = self.index.reconstruct_n(0, self.index.ntotal)
        all_vecs = faiss.revdocs.concatenate((all_vecs, new_vecs)) if all_vecs.size else new_vecs

        # ensure unit length for IP metric
        faiss.normalize_L2(all_vecs)

        new_index = self._build_index_factory(all_vecs.shape[0])
        # IVF / PQ indexes need proper training
        if new_index.is_trained is False:
            sample_sz = min(250_000, all_vecs.shape[0])
            sample = faiss.random_subset(all_vecs, sample_sz)
            new_index.train(sample)

        ids = faiss.IndexIDMap2.numpy_int64_to_idx64(range(all_vecs.shape[0]))
        new_index.add_with_ids(all_vecs, ids)
        self.index = new_index

        # extend metadata
        self.docs.extend(new_texts)
        start_id = len(self.docs) - len(new_texts)
        self._digest_to_id.update(
            {dg: idx for dg, idx in zip(new_digests, range(start_id, start_id + len(new_texts)))}
        )
        print("[RagStore] Rebuild complete – index type now:", self._extract_factory_string(self.index))

    # ---------- utilities --------------------------------------------- #
    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_factory_string(index: faiss.Index) -> str:
        if hasattr(index, "index") and hasattr(index, "string_opts"):
            return index.string_opts  # type: ignore[attr-defined]
        # IndexIDMap -> extract child
        if isinstance(index, faiss.IndexPreTransform):
            return RagStore._extract_factory_string(index.index)  # type: ignore[arg-type]
        if isinstance(index, faiss.IndexIDMap2):
            return RagStore._extract_factory_string(index.index)  # type: ignore[arg-type]
        # Fallback
        return type(index).__name__