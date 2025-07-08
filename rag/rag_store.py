# ============================
# rag/rag_store.py (updated)
# ============================
from __future__ import annotations

import hashlib
import math
import os
import pickle
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


class RagStore:
    """Hierarchy‑aware, auto‑scaling FAISS store."""

    # ---------- tier thresholds & factories --------------------------- #
    _TIER_CFG: List[Tuple[int, str]] = [
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

        # SHA‑256 → doc_id map for dedup
        self._digest_to_id: dict[str, int] = {
            hashlib.sha256(d.encode("utf-8")).hexdigest(): i for i, d in enumerate(self.docs)
        }

        # light in‑memory query log: (timestamp, vec, hit_ids)
        self._query_log: List[Tuple[float, np.ndarray, List[int]]] = []
        self._query_log_max = 5_000

    # ------------------------------------------------------------------ #
    # public API

    def add_documents(self, docs: Sequence[str]) -> None:
        """Embed and add *unique* docs."""
        uniques = [(doc, self._hash(doc)) for doc in docs if self._hash(doc) not in self._digest_to_id]
        if not uniques:
            return

        texts, digests = zip(*uniques)
        embs = self.embed.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)

        new_total = self.index.ntotal + embs.shape[0]
        if self._requires_rebuild(new_total):
            self._rebuild_index(embs, list(texts), list(digests))
            return

        ids = np.arange(len(self.docs), len(self.docs) + len(texts)).astype("int64")
        self.index.add_with_ids(embs, ids)  # type: ignore[arg-type]
        self.docs.extend(texts)
        self._digest_to_id.update(dict(zip(digests, ids)))

    def query(self, text: str, top_k: int = 5, mode: str = "auto") -> List[str]:
        if self.index.ntotal == 0:
            return []

        q_vec = self.embed.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        self._tune_nprobe(mode)
        dists, idxs = self.index.search(q_vec, top_k)  # type: ignore[misc]

        # query‑log for expansion module
        self._log_query(q_vec, idxs[0].tolist())
        return [self.docs[i] for i in idxs[0] if 0 <= i < len(self.docs)]

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))  # type: ignore[arg-type]
        with open(self.index_path.with_suffix(".meta"), "wb") as f:
            pickle.dump(self.docs, f)

    # ---------- helper getters for expansion module ------------------- #
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def is_duplicate(self, doc: str) -> bool:
        return self._hash(doc) in self._digest_to_id

    def get_embeddings(self, ids: Sequence[int]) -> np.ndarray:
        return self.index.reconstruct_n(int(ids[0]), len(ids))  # type: ignore[arg-type]

    # stubbed cluster helpers – single‑index version returns default values
    def list_clusters(self) -> List[int]:
        return [0]

    def cluster_stats(self, _cid: int) -> dict:
        age_days = max(1, int((time.time() - os.path.getmtime(self.index_path)) / 86_400))
        return {"doc_count": self.ntotal(), "age_days": age_days}

    def router_search(self, vec: np.ndarray, top_n: int = 1):
        # single‑cluster fallback: distance 0, id 0
        return np.zeros((1, top_n)), np.zeros((1, top_n), dtype="int64")

    def get_low_recall_queries(self, min_recall: float = 0.3):
        return []

    # ------------------------------------------------------------------ #
    # private

    def _tune_nprobe(self, mode: str):
        if hasattr(self.index, "nprobe"):
            if mode == "breadth":
                self.index.nprobe = min(64, getattr(self.index, "nlist", 1))
            elif mode == "depth":
                self.index.nprobe = 1
            else:
                self.index.nprobe = 16

    def _log_query(self, q_vec: np.ndarray, hit_ids: List[int]):
        self._query_log.append((time.time(), q_vec, hit_ids))
        if len(self._query_log) > self._query_log_max:
            self._query_log.pop(0)

    def _load_or_init(self):
        meta_path = self.index_path.with_suffix(".meta")
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))  # type: ignore[arg-type]
            with open(meta_path, "rb") as f:
                self.docs: List[str] = pickle.load(f)
        else:
            self.index = self._build_index_factory(0)
            self.docs = []

    def _build_index_factory(self, n_vectors_estimate: int):
        factory = self._choose_factory(n_vectors_estimate)
        index = faiss.index_factory(self.dim, factory, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            dummy = faiss.random_distrib(self.dim, 2048)
            faiss.normalize_L2(dummy)
            index.train(dummy)
        return faiss.IndexIDMap2(index)

    def _choose_factory(self, n_vectors: int) -> str:
        for upper, factory in self._TIER_CFG:
            if n_vectors < upper:
                return factory
        return "HNSW32,Flat"

    def _requires_rebuild(self, new_total: int) -> bool:
        current = self._extract_factory_string(self.index)
        desired = self._choose_factory(new_total)
        return current != desired

    def _rebuild_index(self, new_vecs, new_texts: List[str], new_digests: List[str]):
        print("[RagStore] Rebuilding FAISS index for scale jump…")
        old_vecs = self.index.reconstruct_n(0, self.index.ntotal)
        all_vecs = np.concatenate([old_vecs, new_vecs]) if old_vecs.size else new_vecs
        faiss.normalize_L2(all_vecs)
        new_index = self._build_index_factory(all_vecs.shape[0])
        if not new_index.is_trained:
            sample = all_vecs[np.random.choice(all_vecs.shape[0], min(250_000, all_vecs.shape[0]), replace=False)]
            new_index.train(sample)
        ids = np.arange(all_vecs.shape[0]).astype("int64")
        new_index.add_with_ids(all_vecs, ids)
        self.index = new_index
        offset = len(self.docs)
        self.docs.extend(new_texts)
        self._digest_to_id.update({dg: offset + i for i, dg in enumerate(new_digests)})
        print("[RagStore] Rebuild done – index:", self._extract_factory_string(self.index))

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_factory_string(index: faiss.Index) -> str:
        if isinstance(index, faiss.IndexIDMap2):
            return RagStore._extract_factory_string(index.index)  # type: ignore[arg-type]
        if hasattr(index, "string_opts"):
            return index.string_opts  # type: ignore[attr-defined]
        return type(index).__name__

