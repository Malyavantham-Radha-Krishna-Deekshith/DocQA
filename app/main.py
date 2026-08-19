"""FastAPI backend. Deployed on Render; talked to by the Vercel-hosted
vanilla JS/Tailwind frontend over CORS. Every request carries an
X-Session-Id header identifying the caller's isolated document set + chat
history (see app/session_manager.py) — nothing here is shared between
sessions except the OCR/embedding/LLM clients themselves.
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mistralai.client.errors.sdkerror import SDKError
from pydantic import BaseModel

from app.config import settings
from app.session_manager import sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    sweeper = asyncio.create_task(sessions.run_idle_sweeper())
    yield
    sweeper.cancel()


app = FastAPI(title="Document Q&A API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SDKError)
async def mistral_error_handler(request: Request, exc: SDKError):
    # An unhandled exception here would produce a 500 that bypasses
    # CORSMiddleware entirely (it sits outside the middleware stack for
    # unhandled errors), which the browser reports as a misleading CORS
    # failure instead of the real cause. Catching it here keeps the
    # response inside the normal CORS-wrapped path.
    if exc.status_code == 429:
        detail = "The AI service is receiving too many requests right now. Please wait a moment and try again."
    else:
        detail = "The AI service is temporarily unavailable. Please try again in a moment."
    return JSONResponse(status_code=502, content={"detail": detail})


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: str
    is_grounded: bool
    rewritten_query: str


class ProcessResponse(BaseModel):
    documents_processed: int
    chunks_indexed: int


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/documents", response_model=ProcessResponse)
async def process_documents(files: list[UploadFile], x_session_id: str = Header(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > settings.MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {settings.MAX_IMAGES} images per upload.")

    images: list[tuple[bytes, str]] = []
    for f in files:
        ext = (f.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")
        images.append((await f.read(), f.filename))

    session = sessions.get_or_create(x_session_id)
    summary = session.pipeline.process_documents(images)
    return ProcessResponse(**summary)


@app.delete("/api/documents")
def reset_documents(x_session_id: str = Header(...)):
    sessions.reset(x_session_id)
    return {"status": "reset"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest, x_session_id: str = Header(...)):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    session = sessions.get_or_create(x_session_id)
    result = session.pipeline.answer_question(body.question, session.memory)
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        is_grounded=result.is_grounded,
        rewritten_query=result.rewritten_query,
    )
