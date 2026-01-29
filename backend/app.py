import os
from typing import List, Dict, Any
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

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


def load_vectordb_for_department(dept: str) -> Chroma:
    persist_dir = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name = f"kb_{dept}"  # must match your ingest.py collections: kb_hr, kb_finance, kb_it, kb_general

    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def retrieve_context(vectordb: Chroma, query: str, k: int = 5) -> List[Dict[str, Any]]:
    docs = vectordb.similarity_search(query, k=k)
    return [{"source": d.metadata.get("source", "unknown_source"), "text": d.page_content} for d in docs]


def format_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {c['source']}\n{c['text']}")
    return "\n\n".join(parts)


# ---------------- App ----------------

app = FastAPI(title="Enterprise Real-Time RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory (last N messages per session)
MAX_TURNS = 8
CHAT_STORE = defaultdict(lambda: deque(maxlen=MAX_TURNS))


def format_history(session_id: str) -> str:
    turns = CHAT_STORE[session_id]
    if not turns:
        return ""

    lines = []
    for t in turns:
        if t["role"] == "user":
            lines.append(f"User: {t['content']}")
        else:
            lines.append(f"Assistant: {t['content']}")
    return "\n".join(lines)


# Load department vector DBs once
DEPARTMENTS = ["hr", "finance", "it", "general"]
VDBS = {d: load_vectordb_for_department(d) for d in DEPARTMENTS}

# LLM once
LLM = ChatOpenAI(
    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    temperature=0,
    streaming=True,
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
    # 1) route to department
    dept = route_department(req.message)
    vectordb = VDBS.get(dept, VDBS["general"])

    # 2) retrieve context from that department KB
    chunks = retrieve_context(vectordb, req.message, k=req.k)
    context_text = format_context(chunks)

    # 3) include memory
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

    async def event_generator():
        full_answer = ""
        try:
            async for chunk in LLM.astream(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            ):
                token = chunk.content or ""
                if token:
                    full_answer += token
                    yield f"data: {token}\n\n"

            # Save to memory after completion
            CHAT_STORE[req.session_id].append({"role": "user", "content": req.message})
            CHAT_STORE[req.session_id].append({"role": "assistant", "content": full_answer})

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
