"""Top-level orchestrator wiring OCR -> normalization -> chunking ->
embeddings -> FAISS -> retrieval -> guardrails -> LLM together
(requirement 13's end-to-end flow). This is the module the API layer talks
to. OCR/embedder/LLM clients are injected (shared across sessions by the
caller); the store is per-session and in-memory only — nothing is persisted
to disk, since each session gets its own isolated index.
"""
from dataclasses import dataclass
from typing import List, Tuple

from app.config import settings
from app.chunking.structure_chunker import chunk_document
from app.document_processing.normalizer import normalize_markdown
from app.embeddings.embedder import Embedder
from app.guardrails.grounding import format_sources
from app.llm.mistral_llm import MistralLLMClient
from app.memory.session_memory import SessionMemory
from app.ocr.mistral_ocr import MistralOCRClient
from app.retrieval.retriever import Retriever
from app.vectorstore.faiss_store import FaissStore


@dataclass
class AnswerResult:
    answer: str
    sources: str
    is_grounded: bool
    rewritten_query: str


class DocumentQAPipeline:
    def __init__(self, ocr: MistralOCRClient, embedder: Embedder, llm: MistralLLMClient, store: FaissStore):
        self._ocr = ocr
        self._embedder = embedder
        self._llm = llm
        self._store = store
        self._retriever = Retriever(self._embedder, self._store, self._llm)

    def process_documents(self, images: List[Tuple[bytes, str]]) -> dict:
        """images: list of (image_bytes, filename). Returns a summary dict."""
        ocr_results = self._ocr.process_images(images)

        total_chunks = 0
        for ocr_result in ocr_results:
            for page in ocr_result.pages:
                page.markdown = normalize_markdown(page.markdown)

            chunks = chunk_document(
                ocr_result,
                max_tokens=settings.CHUNK_SIZE_TOKENS,
                overlap_ratio=settings.DEFAULT_OVERLAP_RATIO,
            )
            if not chunks:
                continue

            embeddings = self._embedder.embed_texts([c.text for c in chunks])
            self._store.add(chunks, embeddings)
            total_chunks += len(chunks)

        return {
            "documents_processed": len(ocr_results),
            "chunks_indexed": total_chunks,
        }

    def answer_question(self, question: str, memory: SessionMemory) -> AnswerResult:
        # Runs before checking whether anything's been uploaded — a greeting
        # ("hai", "thanks") is just as likely to open a conversation as
        # follow one, and shouldn't need a document indexed to get a normal
        # reply instead of the cold not-found guardrail message.
        result = self._retriever.retrieve(question, memory)

        if result.is_conversational:
            memory.add_turn(question, result.conversational_reply)
            return AnswerResult(
                answer=result.conversational_reply,
                sources="",
                is_grounded=False,
                rewritten_query=result.rewritten_query,
            )

        if self._store.is_empty:
            answer = settings.NOT_FOUND_MESSAGE
            memory.add_turn(question, answer)
            return AnswerResult(
                answer=answer,
                sources="",
                is_grounded=False,
                rewritten_query=result.rewritten_query,
            )

        if not result.is_relevant or not result.chunks:
            answer = settings.NOT_FOUND_MESSAGE
            memory.add_turn(question, answer)
            return AnswerResult(
                answer=answer,
                sources="",
                is_grounded=False,
                rewritten_query=result.rewritten_query,
            )

        answer = self._llm.generate_answer(result.rewritten_query, result.chunks, broad_overview=result.is_broad_overview)
        memory.add_turn(question, answer)

        # Retrieval found *something* similar enough to pass the relevance
        # gate, but the model itself can still decide none of it actually
        # answers the question and fall back to NOT_FOUND_MESSAGE (system
        # prompt rule 5). In that case sources shouldn't be shown — citing
        # sources next to "couldn't find this" reads as contradictory.
        actually_grounded = answer.strip() != settings.NOT_FOUND_MESSAGE
        sources = format_sources(result.chunks) if actually_grounded else ""

        return AnswerResult(
            answer=answer,
            sources=sources,
            is_grounded=actually_grounded,
            rewritten_query=result.rewritten_query,
        )
