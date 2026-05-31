import os
from typing import List, Dict, Any
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from backend.router import route_department


load_dotenv()

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

Settings.embed_model = OpenAIEmbedding()


def load_index_for_department(dept: str) -> VectorStoreIndex:
    persist_dir = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name = f"kb_{dept}"

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


# ---------------- App ----------------

app = FastAPI(title="Enterprise Real-Time RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_TURNS = 8
CHAT_STORE = defaultdict(lambda: deque(maxlen=MAX_TURNS))


def format_history(session_id: str) -> str:
    turns = CHAT_STORE[session_id]
    if not turns:
        return ""
    lines = []
    for t in turns:
        prefix = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {t['content']}")
    return "\n".join(lines)


DEPARTMENTS = ["hr", "finance", "it", "general"]
INDEXES = {d: load_index_for_department(d) for d in DEPARTMENTS}

LLM = OpenAI(
    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    temperature=0,
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    k: int = 5


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat/reset")
def chat_reset(req: ResetRequest):
    CHAT_STORE.pop(req.session_id, None)
    return {"status": "reset", "session_id": req.session_id}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    dept = route_department(req.message)
    index = INDEXES.get(dept, INDEXES["general"])

    chunks = retrieve_context(index, req.message, k=req.k)
    context_text = format_context(chunks)
    history_text = format_history(req.session_id)

    user_prompt = f"""Department selected: {dept.upper()}

Chat history (most recent last):
{history_text if history_text else "[no prior messages]"}

Context:
{context_text}

User question:
{req.message}

Instructions:
- Use ONLY the context above to answer policy/FAQ questions.
- If the context does not contain enough information, say:
  "I don't know based on the provided company knowledge."
  and DO NOT list sources.
- If you answer, include:
Sources:
- <source file>
(only sources that support the answer)
"""

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]

    async def event_generator():
        full_answer = ""
        try:
            async for chunk in await LLM.astream_chat(messages):
                token = chunk.delta or ""
                if token:
                    full_answer += token
                    yield f"data: {token}\n\n"

            CHAT_STORE[req.session_id].append({"role": "user", "content": req.message})
            CHAT_STORE[req.session_id].append({"role": "assistant", "content": full_answer})

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
