"""RAG-level evaluation metrics (requirement 15). Interfaces only for now —
to be filled in once we have a labeled question/answer eval set, so we can
compare chunking/overlap presets (app.config.settings.OVERLAP_PRESETS) and
OCR representations against each other.
"""


def retrieval_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int) -> float:
    """Top-K retrieval accuracy: did a relevant chunk appear in the top K results."""
    raise NotImplementedError("TODO: implement once a labeled retrieval eval set exists")


def context_relevance_score(question: str, retrieved_chunks: list[str]) -> float:
    raise NotImplementedError("TODO: define via LLM-judge or human-labeled relevance")


def answer_accuracy(predicted_answer: str, gold_answer: str) -> float:
    raise NotImplementedError("TODO: define exact-match / semantic-match scoring")


def groundedness_score(answer: str, context_chunks: list[str]) -> float:
    """Fraction of answer claims that are directly supported by the
    retrieved context (vs. unsupported/fabricated)."""
    raise NotImplementedError("TODO: implement via LLM-judge or NLI-based entailment check")


def hallucination_rate(answers: list[str], context_chunks_per_answer: list[list[str]]) -> float:
    raise NotImplementedError("TODO: aggregate groundedness_score across an eval set")
