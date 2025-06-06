# AATIS-SKMS
Experimental AATIS System. Smart Knowledge Management Experimentation. 

# Smart Methodology Knowledge System Plus (SMKS+)
*Slimline AATIS edition with self expanding research & knowledge management*

---

## Goals

* **Deep Research Automation** fill knowledge gaps through curated web search & summarisation.  
* **Multiâ€‘page Markdown Generation** produce, save & categorise rich documentation.  
* **Persistent Knowledge Base** store human & AI‘generated docs; enable fast Q&A via RAG/FAISS.  
* **User in the Loop Control**  accept feedback, refine files, remember preferences.  
* **Lightweight & Secure** no active pentesting tools; focuses on cognition & content.

---

## High Level Architecture

```mermaid
graph TD
  U(User) -->|Prompt / Feedback| RT(TaskRouter)
  RT --> MT[MethodologyÂ Team]
  RT --> WR[WebÂ ResearchÂ Team]
  RT --> QA[Q&AÂ Team]
  RT --> FM[FileÂ Mgr]
  WR -->|Docs + metadata| ING[IngestionÂ Agent]
  MT -->|Markdown drafts| FM
  ING --> IDX[IndexÂ Builder]
  FM --> ING
  IDX --> VDB((FAISSÂ Vector DB))
  VDB --> QA
  QA -->|Answers| U
  MT -->|Gap queries| WR
  QA -. update .-> MT
  U -->|File uploads (pdf/txt)| FM
```

---

## Agent Teams & Responsibilities

| Team | Agents | Purpose | Key Tools |
|------|--------|---------|-----------|
| **Core Manager** | `TaskRouter` (decides which team handles a sub task), `ResourceScheduler` (rateâ€‘limits API / net calls) | Governance & orchestration | AutoGen `GroupChatManager`, pydantic config |
| **Methodology** | `DocGenerator`, `GapAnalyzer` | Draft methodology pages, detect missing topics, request research | OpenAI models (tempÂ 0.6), Markdown templates |
| **Web Research** | `SearchAgent`, `PageReader`, `SummaryAgent` | Query internet, scrape content, produce concise notes | DuckDuckGo API, newspaper3k, trafilatura |
| **RAG Pipeline** | `IngestionAgent`, `IndexBuilder`, `Retriever` (`Librarian`) | Clean & chunk docs, build FAISS index, answer queries | LangChain, FAISS, sentenceâ€‘transformers |
| **File Management** | `FileOrganizer`, `VersionController` | Save docs to `/knowledge/{topic}/vN`, maintain metadata JSON, commit to git if configured | Python `pathlib`, `gitpython` |
| **Feedback & Editing** | `FeedbackListener`, `DocEditor` | Log user corrections, patch files, trigger reâ€‘index | Simstring diff, MarkdownÂ it |
| **Memory & State** | SharedVectorMemory (FAISS), RelationalMemory (SQLite) | Store embeddings & metadata | FAISS, peewee ORM |

---

## Data Flow Detail

1. **User Prompt**  `TaskRouter`  
2. **Router** labels task (`write`, `research`, `question`, `feedback`).  
3. **Methodology** writes draft or detects *knowledge gap*.  
4. **Gap** triggers **Web Research** searches, scrapes, summarises.  
5. **Ingestion** normalises new docs **IndexBuilder** updates FAISS.  
6. **DocGenerator** finalises Markdown **FileOrganizer** saves with formatter.  
7. **Retriever** services Q&A queries using hybrid (keyword + cosine) search.  
8. **User Feedback** updates docs ingest & index.

---

## Component Specs

### TaskRouter (Core)
```python
class TaskRouter(Agent):
    def route(self, task):
        if task.type in {"ask", "lookup"}:
            return "Retriever"
        if "research" in task.tags or "gap" in task.tags:
            return "WebResearch"
        # fallthrough ...
```
*Maintains context map: task ID, thread state.*

### GapAnalyzer
* Embeds each draft section.  
* Searches FAISS for **< sim 0.60** segments â†’ missing.  
* Emits `research_needed` tasks with keywords.

### 5.3 SearchAgent
* Queries DDG top n links.  
* Filters by domain allow list / date.  
* Hands URLs to `PageReader`.

### IndexBuilder
```bash
python ingest.py --input ./knowledge --out ./db/faiss
```
* Uses `text_splitter` (chunk 800/200 overlap).  
* Stores `{uuid, path, chunk_id, tags}` in SQLite.

*(Further class templates in Appendix A).*

---

## Directory Layout

```
SMKS/
â”œâ”€â”€ core/
â”‚   â””â”€â”€ router.py
â”œâ”€â”€ agents/
â”‚   â”œâ”€â”€ methodology/
â”‚   â”œâ”€â”€ web_research/
â”‚   â””â”€â”€ rag/
â”œâ”€â”€ knowledge/
â”‚   â””â”€â”€ <topic>/
â”‚       â”œâ”€â”€ v1/
â”‚       â”‚   â””â”€â”€ index.md
â”‚       â””â”€â”€ v2/ ...
â”œâ”€â”€ db/
â”‚   â”œâ”€â”€ faiss/
â”‚   â””â”€â”€ meta.sqlite
â””â”€â”€ config.yaml
```

---

## AutoGen GroupChat Snippet

```python
from autogen import Agent, GroupChat, GroupChatManager

router = TaskRouter(...)
librarian = Retriever(...)
docgen = DocGenerator(...)
search = SearchAgent(...)

gc = GroupChat(agents=[router, librarian, docgen, search],
               manager=GroupChatManager(max_rounds=12))
response = gc.chat("Write initial reconnaissance methodology for Azure AD")
```

---

## User Facing CLI

```bash
$ smks new "Internal Pentest on Windows Server 2019"
$ smks ask "How to avoid Azure ATP detection?"
$ smks feedback file.md --line 84 "This technique is outdated"
```

All commands trigger underlying agents via argparse â†’ RPC.

---

## Security & Ethics

* **No code execution** all content strictly text.  
* **Content provenance** stored (`source_url`, timestamp).  
* **Rateâ€‘limiting** to respect website TOS.  
* **User approval** before persisting scraped excerpts.

---

## Future Work

* plugin for browser? save information on sites  
* multitenant workspace isolation  
* optional vision capability for diagram OCR

---

### Appendix A: Agent Class Skeletons
*(truncated for brevity see `/agents/` templates)*
