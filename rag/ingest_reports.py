import glob
from rag.rag_store import RagStore

def refresh_reports(reports_dir="reports/"):
    store = RagStore()
    files = glob.glob(f"{reports_dir}/*.md")
    if not files:
        print("No reports found.")
        return

    docs = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            docs.append(f.read())

    store.add_documents(docs)
    store.save()
    print(f"Ingested {len(docs)} reports into RAG store.")

if __name__ == "__main__":
    refresh_reports()
