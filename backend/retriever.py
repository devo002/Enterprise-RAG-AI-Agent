import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def get_vectordb(persist_dir: str, collection_name: str = "enterprise_kb") -> Chroma:
    """Load an existing Chroma vector DB from disk."""
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma persist directory not found: {persist_path.resolve()}\n"
            f"Run ingestion first: python backend/ingest.py"
        )

    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vectordb


def search(
    query: str,
    k: int = 5,
    persist_dir: str = "chroma_db",
    collection_name: str = "enterprise_kb",
) -> List[Tuple[str, str]]:
    """
    Return top-k results as (source, chunk_text).
    Source comes from metadata stored during ingestion.
    """
    vectordb = get_vectordb(persist_dir=persist_dir, collection_name=collection_name)

    results = vectordb.similarity_search(query, k=k)

    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "unknown_source")
        formatted.append((source, doc.page_content))
    return formatted


def main():
    load_dotenv()

    # Optional: allow overrides via .env
    persist_dir = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "enterprise_kb")

    print("Chroma DB:", Path(persist_dir).resolve())
    print("Collection:", collection_name)
    print("-" * 60)

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        hits = search(
            query=q,
            k=5,
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        print(f"\nTop matches for: {q!r}\n")
        for i, (src, text) in enumerate(hits, start=1):
            preview = text.strip().replace("\n", " ")
            if len(preview) > 350:
                preview = preview[:350] + "..."
            print(f"{i}. Source: {src}")
            print(f"   Chunk:  {preview}")
            print()

    print("\nBye!")


if __name__ == "__main__":
    main()
