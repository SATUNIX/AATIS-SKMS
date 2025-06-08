import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss

class RagStore:
    def __init__(self, index_path="rag/index.faiss", embed_model="all-MiniLM-L6-v2"):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.index_path = index_path
        self.embed = SentenceTransformer(embed_model)
        dim = self.embed.get_sentence_embedding_dimension()

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(index_path + ".meta", "rb") as f:
                self.docs = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.docs = []

    def add_documents(self, docs: list[str]):
        embs = self.embed.encode(docs, convert_to_numpy=True)
        self.index.add(embs)
        self.docs.extend(docs)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".meta", "wb") as f:
            pickle.dump(self.docs, f)

    def query(self, text: str, top_k=5):
        q_emb = self.embed.encode([text], convert_to_numpy=True)
        dists, idxs = self.index.search(q_emb, top_k)
        return [self.docs[i] for i in idxs[0] if i < len(self.docs)]
