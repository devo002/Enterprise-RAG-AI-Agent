import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def get_index(persist_dir: str, collection_name: str = "enterprise_kb") -> VectorStoreIndex:
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma persist directory not found: {persist_path.resolve()}\n"
            f"Run ingestion first: python backend/ingest.py"
        )

    Settings.embed_model = OpenAIEmbedding()
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def search(
    query: str,
    k: int = 5,
    persist_dir: str = "chroma_db",
    collection_name: str = "enterprise_kb",
) -> List[Tuple[str, str]]:
    index = get_index(persist_dir=persist_dir, collection_name=collection_name)
    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)
    return [(n.metadata.get("source", "unknown_source"), n.text) for n in nodes]


def main():
    load_dotenv()

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

        hits = search(query=q, k=5, persist_dir=persist_dir, collection_name=collection_name)

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
