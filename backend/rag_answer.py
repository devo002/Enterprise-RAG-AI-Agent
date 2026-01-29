import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma


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



def load_vectordb(persist_dir: str, collection_name: str) -> Chroma:
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma DB not found at: {persist_path.resolve()}\n"
            f"Run ingestion first: python backend/ingest.py"
        )

    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def retrieve_context(vectordb: Chroma, query: str, k: int = 5) -> List[Dict[str, Any]]:
    docs = vectordb.similarity_search(query, k=k)
    results = []
    for d in docs:
        results.append(
            {
                "source": d.metadata.get("source", "unknown_source"),
                "text": d.page_content,
            }
        )
    return results


def format_context(chunks: List[Dict[str, Any]]) -> str:
    # Create a readable context block with sources
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {c['source']}\n{c['text']}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 5) -> str:
    persist_dir = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "enterprise_kb")

    vectordb = load_vectordb(persist_dir, collection_name)
    chunks = retrieve_context(vectordb, question, k=k)

    context_text = format_context(chunks)

    llm = ChatOpenAI(
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

    resp = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    return resp.content


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
