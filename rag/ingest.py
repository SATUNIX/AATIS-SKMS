# ingest.py
import glob
import os
from rag.rag_store import RagStore

def refresh_store(reports_dir="reports/", web_content_dir="web_content/"):
    """
    Ingest both .md reports and .txt web_content into the RAG index.
    """
    store = RagStore()

    # find all markdown reports
    report_files = glob.glob(os.path.join(reports_dir, "*.md"))
    # find all web content text files
    web_files    = glob.glob(os.path.join(web_content_dir, "*.txt"))

    all_files = report_files + web_files
    if not all_files:
        print("No reports or web content files found.")
        return

    docs = []
    for path in all_files:
        with open(path, "r", encoding="utf-8") as f:
            docs.append(f.read())

    store.add_documents(docs)
    store.save()
    print(
        f"Ingested {len(report_files)} reports "
        f"and {len(web_files)} web content files "
        "into RAG store."
    )

if __name__ == "__main__":
    refresh_store()
