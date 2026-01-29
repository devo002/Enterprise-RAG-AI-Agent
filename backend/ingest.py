import os
import glob
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


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
        # department is the immediate folder under knowledge_base
        # e.g. knowledge_base/hr/remote_work_policy.md -> dept = "hr"
        try:
            dept = path.relative_to(kb_path).parts[0].lower()
        except Exception:
            dept = "general"

        text = path.read_text(encoding="utf-8", errors="ignore")
        dept_docs[dept].append(
            Document(
                page_content=text,
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
        import shutil
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    print(f"[1/4] Loading knowledge base from: {Path(KB_DIR).resolve()}")
    dept_docs = load_department_docs(KB_DIR)
    print(f"      Departments found: {list(dept_docs.keys())}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    embeddings = OpenAIEmbeddings()

    print("[2/4] Splitting + indexing per department...")
    for dept, docs in dept_docs.items():
        chunks = splitter.split_documents(docs)
        collection_name = f"kb_{dept}"

        print(f"   - Dept: {dept} | files={len(docs)} | chunks={len(chunks)} | collection={collection_name}")

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=collection_name,
        )
        vectordb.persist()

    print(f"[3/4] Done ✅ Saved collections into: {Path(PERSIST_DIR).resolve()}")
    print("[4/4] Next: update backend router to choose department per question.")


if __name__ == "__main__":
    main()
