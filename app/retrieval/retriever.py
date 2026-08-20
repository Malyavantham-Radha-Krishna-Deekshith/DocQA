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
    is_broad_overview: bool = False
    is_conversational: bool = False
    conversational_reply: str = ""


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
        understanding = self._llm.understand_query(
            question, memory.as_context_string(), self._store.document_filenames
        )

        if understanding.is_conversational:
            return RetrievalResult(
                rewritten_query=understanding.rewritten_query,
                chunks=[],
                is_relevant=False,
                is_conversational=True,
                conversational_reply=understanding.conversational_reply,
            )

        if understanding.is_broad_overview:
            # Bypass similarity search entirely: a vague "give me everything"
            # question doesn't meaningfully rank documents by relevance, so
            # top-k would arbitrarily favor whichever ones happen to embed
            # closest to the vague phrasing. Include every document instead.
            chunks = self._store.all_metadata()[: settings.MAX_BROAD_CONTEXT_CHUNKS]
            return RetrievalResult(
                rewritten_query=understanding.rewritten_query,
                chunks=chunks,
                is_relevant=bool(chunks),
                is_broad_overview=True,
            )

        query_embedding = self._embedder.embed_query(understanding.rewritten_query)
        chunks = self._store.search(query_embedding, top_k=top_k)

        top_score = chunks[0]["score"] if chunks else 0.0
        relevant = is_relevant(top_score, threshold)

        return RetrievalResult(
            rewritten_query=understanding.rewritten_query,
            chunks=chunks if relevant else [],
            is_relevant=relevant,
        )
