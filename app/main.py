import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .generator import GrokError, rag_answer
from .ingest import DOCUMENTS_DIR, ensure_indexed
from .retriever import HybridRetriever

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./data/chroma")

retriever = HybridRetriever(persist_dir=VECTOR_STORE_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(ensure_indexed, retriever, DOCUMENTS_DIR)
    yield


app = FastAPI(title="Czech Forecast RAG", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
async def health():
    stats = retriever.stats()
    return {"status": "ok", "documents_dir": DOCUMENTS_DIR, **stats}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    top_k: int = 5


class ChatSource(BaseModel):
    page: Optional[int] = None
    source: Optional[str] = None
    type: str = "text"
    score: float = 0.0


class ChatResponse(BaseModel):
    response: str
    sources: List[ChatSource] = []
    used_llm: bool = False
    error: Optional[str] = None


@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    top_k = max(1, min(req.top_k, 20))
    hits = await run_in_threadpool(retriever.retrieve, question, top_k)
    contexts = [doc for _, doc in hits]
    history = [{"role": m.role, "content": m.content} for m in (req.history or [])]

    sources = [
        ChatSource(
            page=(doc.get("meta") or {}).get("page"),
            source=(doc.get("meta") or {}).get("source"),
            type=(doc.get("meta") or {}).get("type", "text"),
            score=round(float(score), 4),
        )
        for score, doc in hits
    ]

    try:
        answer = await run_in_threadpool(rag_answer, question, contexts, history)
        return ChatResponse(response=answer, sources=sources, used_llm=True)
    except GrokError as exc:
        logger.warning("Generation failed: %s", exc)
        return ChatResponse(
            response="I could not generate an answer because the language model API failed.",
            sources=sources,
            used_llm=False,
            error=str(exc),
        )