"""Per-session state for the FastAPI backend.

The OCR/embedding/LLM clients are heavy (a loaded sentence-transformers
model, HTTP clients) but stateless, so one instance of each is shared across
every visitor. Only the lightweight, mutable parts — a session's indexed
chunks (FaissStore) and its chat history (SessionMemory) — are kept
per-session, in memory, isolated from every other session. Idle sessions are
swept periodically so abandoned browser tabs don't grow memory unbounded.
"""
import asyncio
import time
from dataclasses import dataclass, field

from app.config import settings
from app.embeddings.embedder import Embedder
from app.llm.mistral_llm import MistralLLMClient
from app.memory.session_memory import SessionMemory
from app.ocr.mistral_ocr import MistralOCRClient
from app.pipeline import DocumentQAPipeline
from app.vectorstore.faiss_store import FaissStore


class SharedClients:
    """Lazily-created, process-wide singletons."""

    _ocr: MistralOCRClient | None = None
    _embedder: Embedder | None = None
    _llm: MistralLLMClient | None = None

    @classmethod
    def ocr(cls) -> MistralOCRClient:
        if cls._ocr is None:
            cls._ocr = MistralOCRClient()
        return cls._ocr

    @classmethod
    def embedder(cls) -> Embedder:
        if cls._embedder is None:
            cls._embedder = Embedder()
        return cls._embedder

    @classmethod
    def llm(cls) -> MistralLLMClient:
        if cls._llm is None:
            cls._llm = MistralLLMClient()
        return cls._llm


@dataclass
class SessionState:
    store: FaissStore
    memory: SessionMemory
    pipeline: DocumentQAPipeline
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        session = self._sessions.get(session_id)
        if session is None:
            session = self._new_session()
            self._sessions[session_id] = session
        session.touch()
        return session

    def reset(self, session_id: str) -> SessionState:
        session = self._new_session()
        self._sessions[session_id] = session
        return session

    def _new_session(self) -> SessionState:
        embedder = SharedClients.embedder()
        store = FaissStore(embedder.dimension)
        return SessionState(
            store=store,
            memory=SessionMemory(),
            pipeline=DocumentQAPipeline(SharedClients.ocr(), embedder, SharedClients.llm(), store),
        )

    def sweep_idle(self) -> None:
        cutoff = time.time() - settings.SESSION_IDLE_SECONDS
        idle_ids = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in idle_ids:
            del self._sessions[sid]

    async def run_idle_sweeper(self) -> None:
        while True:
            await asyncio.sleep(settings.SESSION_SWEEP_INTERVAL_SECONDS)
            self.sweep_idle()


sessions = SessionManager()
