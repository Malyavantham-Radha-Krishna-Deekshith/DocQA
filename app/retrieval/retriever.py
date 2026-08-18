"""Query pipeline (requirement 7): session memory -> query understanding /
rewriting -> embedding -> FAISS retrieval -> relevance check.
"""
from dataclasses import dataclass
from typing import List

from app.config import settings
from app.embeddings.embedder import Embedder
from app.guardrails.grounding import is_relevant
from app.llm.mistral_llm import MistralLLMClient
from app.memory.session_memory import SessionMemory
from app.vectorstore.faiss_store import FaissStore


@dataclass
class RetrievalResult:
    rewritten_query: str
    chunks: List[dict]
    is_relevant: bool


class Retriever:
    def __init__(self, embedder: Embedder, store: FaissStore, llm: MistralLLMClient):
        self._embedder = embedder
        self._store = store
        self._llm = llm

    def retrieve(
        self,
        question: str,
        memory: SessionMemory,
        top_k: int = settings.TOP_K,
        threshold: float = settings.RELEVANCE_THRESHOLD,
    ) -> RetrievalResult:
        rewritten = self._llm.rewrite_query(question, memory.as_context_string())

        query_embedding = self._embedder.embed_query(rewritten)
        chunks = self._store.search(query_embedding, top_k=top_k)

        top_score = chunks[0]["score"] if chunks else 0.0
        relevant = is_relevant(top_score, threshold)

        return RetrievalResult(
            rewritten_query=rewritten,
            chunks=chunks if relevant else [],
            is_relevant=relevant,
        )
