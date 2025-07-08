# ============================
# ingest.py (updated)
# ============================
import glob
import os
from rag.rag_store import RagStore


def refresh_store(reports_dir: str = "reports/", web_content_dir: str = "web_content/") -> None:
    """Ingest all markdown and raw‑text sources into the RAG store.

    The function is idempotent; re‑running it simply appends any new
    documents that are not already present.  Duplicate detection is
    performed on SHA‑256 hashes before any embedding work, so the cost
    of repeated runs is negligible.
    """

    store = RagStore()

    # ---------- discover files ----------------------------------------- #
    report_files = glob.glob(os.path.join(reports_dir, "*.md"))
    web_files = glob.glob(os.path.join(web_content_dir, "*.txt"))
    all_files = report_files + web_files

    if not all_files:
        print("No reports or web content files found.")
        return

    # ---------- read & filter duplicates ------------------------------- #
    new_docs = []
    for path in all_files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if store.is_duplicate(text):
            continue
        new_docs.append(text)

    if not new_docs:
        print("No new unique documents to ingest.")
        return

    # ---------- embed & add ------------------------------------------- #
    store.add_documents(new_docs)
    store.save()

    print(
        f"Ingested {len(new_docs):,} new documents ("
        f"{len(report_files):,} reports, {len(web_files):,} web snippets).  "
        f"Store now contains {store.ntotal():,} vectors."
    )


if __name__ == "__main__":
    refresh_store()
