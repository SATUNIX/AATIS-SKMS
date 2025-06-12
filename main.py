import sys
import asyncio

from agents.generate_agent import run_generate
from agents.generate_agent_large import run_generate as run_generate_large
from agents.qa_agent import run_qa
from rag.ingest import refresh_store

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [generate|generate-large|ask|refresh] <topic/question>")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "generate":
        if len(sys.argv) < 3:
            print("Error: Missing topic.")
            sys.exit(1)
        topic = " ".join(sys.argv[2:])
        asyncio.run(run_generate(topic))

    elif cmd == "generate-large":
        if len(sys.argv) < 3:
            print("Error: Missing topic.")
            sys.exit(1)
        topic = " ".join(sys.argv[2:])
        asyncio.run(run_generate_large(topic))

    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Error: Missing question.")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        asyncio.run(run_qa(question))

    elif cmd == "refresh":
        refresh_store()

    else:
        print(f"Unknown command: {cmd}")
        print("Valid commands: generate, generate-large, ask, refresh")
        sys.exit(1)

if __name__ == "__main__":
    main()
