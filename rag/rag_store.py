# rag/rag_store.py

import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore

class RagStore:
    def __init__(self, index_path="rag/index.faiss", embed_model="all-MiniLM-L6-v2"):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.index_path = index_path
        self.embed = SentenceTransformer(embed_model)
        dim = self.embed.get_sentence_embedding_dimension()

        if os.path.exists(index_path):
            # Load existing index
            self.index = faiss.read_index(index_path)  # type: ignore
            with open(index_path + ".meta", "rb") as f:
                self.docs = pickle.load(f)
        else:
            # Create new index
            self.index = faiss.IndexFlatL2(dim)  # type: ignore
            self.docs = []

    def add_documents(self, docs: list[str]):
        embs = self.embed.encode(docs, convert_to_numpy=True)
        self.index.add(embs)  # type: ignore
        self.docs.extend(docs)

    def save(self):
        faiss.write_index(self.index, self.index_path)  # type: ignore
        with open(self.index_path + ".meta", "wb") as f:
            pickle.dump(self.docs, f)

    def query(self, text: str, top_k=5):
        # Early exit if no documents
        if not self.docs or getattr(self.index, "ntotal", 0) == 0:
            return []
        q_emb = self.embed.encode([text], convert_to_numpy=True)
        # Suppress stub errors: faiss.Python API is dynamic
        dists, idxs = self.index.search(q_emb, top_k)  # type: ignore
        return [self.docs[i] for i in idxs[0] if i < len(self.docs)]
