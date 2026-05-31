import os
import glob
import shutil
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def load_department_docs(kb_dir: str) -> dict[str, list[Document]]:
    """
    Loads .md/.txt under knowledge_base/<department>/...
    Returns: { "hr": [Document...], "finance": [...], ... }
    """
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_path.resolve()}")

    files = glob.glob(str(kb_path / "**" / "*.md"), recursive=True)
    files += glob.glob(str(kb_path / "**" / "*.txt"), recursive=True)

    if not files:
        raise FileNotFoundError(f"No .md or .txt files found inside: {kb_path.resolve()}")

    dept_docs: dict[str, list[Document]] = defaultdict(list)

    for f in files:
        path = Path(f)
        try:
            dept = path.relative_to(kb_path).parts[0].lower()
        except Exception:
            dept = "general"

        text = path.read_text(encoding="utf-8", errors="ignore")
        dept_docs[dept].append(
            Document(
                text=text,
                metadata={
                    "source": str(path.as_posix()),
                    "department": dept,
                },
            )
        )

    return dept_docs


def main():
    load_dotenv()

    KB_DIR = os.getenv("KB_DIR", "knowledge_base")
    PERSIST_DIR = os.getenv("CHROMA_DIR", "chroma_db")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env (project root).")

    reset = os.getenv("RESET_CHROMA", "false").lower() == "true"
    if reset and Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    Settings.embed_model = OpenAIEmbedding()

    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=150)

    print(f"[1/4] Loading knowledge base from: {Path(KB_DIR).resolve()}")
    dept_docs = load_department_docs(KB_DIR)
    print(f"      Departments found: {list(dept_docs.keys())}")

    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

    print("[2/4] Splitting + indexing per department...")
    for dept, docs in dept_docs.items():
        collection_name = f"kb_{dept}"
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            transformations=[splitter],
        )
        node_count = len(index.docstore.docs)
        print(f"   - Dept: {dept} | files={len(docs)} | chunks={node_count} | collection={collection_name}")

    print(f"[3/4] Done ✅ Saved collections into: {Path(PERSIST_DIR).resolve()}")
    print("[4/4] Next: update backend router to choose department per question.")


if __name__ == "__main__":
    main()
