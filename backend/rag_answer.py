import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


SYSTEM_PROMPT = """You are an enterprise knowledge assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information, say:
"I don't know based on the provided company knowledge."
In that case, DO NOT list any sources.

Always be concise and policy-accurate.

If you provide an answer, include a section at the end:

Sources:
- <source file name>

Only list sources that directly support the answer.
"""


def load_index(persist_dir: str, collection_name: str) -> VectorStoreIndex:
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma DB not found at: {persist_path.resolve()}\n"
            f"Run ingestion first: python backend/ingest.py"
        )

    Settings.embed_model = OpenAIEmbedding()
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def retrieve_context(index: VectorStoreIndex, query: str, k: int = 5) -> List[Dict[str, Any]]:
    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)
    return [{"source": n.metadata.get("source", "unknown_source"), "text": n.text} for n in nodes]


def format_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {c['source']}\n{c['text']}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 5) -> str:
    persist_dir = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "enterprise_kb")

    index = load_index(persist_dir, collection_name)
    chunks = retrieve_context(index, question, k=k)
    context_text = format_context(chunks)

    llm = OpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    user_prompt = f"""Context:
{context_text}

User question:
{question}

Instructions:
- Use ONLY the context above.
- If not enough info, say you don't know based on the provided company knowledge.
- Provide a clear answer.
- Then include:
Sources:
- <source file 1>
- <source file 2>
(using only sources that appeared in the context)
"""

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]

    resp = llm.chat(messages)
    return resp.message.content


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in a .env file in the project root."
        )

    print("RAG Answer (grounded in enterprise_kb)")
    print("Type 'exit' to quit.")
    print("-" * 60)

    while True:
        q = input("\nAsk a question: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        try:
            ans = answer_question(q, k=5)
            print("\n" + ans + "\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
