"""Mistral chat client for query understanding and grounded answer generation."""
import json
from dataclasses import dataclass

from mistralai.client import Mistral

from app.config import settings
from app.guardrails.grounding import build_grounding_system_prompt, build_context_block, NOT_FOUND_MESSAGE


@dataclass
class QueryUnderstanding:
    rewritten_query: str
    is_broad_overview: bool


class MistralLLMClient:
    def __init__(self, api_key: str = settings.MISTRAL_API_KEY, model: str = settings.MISTRAL_LLM_MODEL):
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self._client = Mistral(api_key=api_key)
        self._model = model

    def understand_query(
        self, question: str, memory_context: str, document_names: list[str] | None = None
    ) -> QueryUnderstanding:
        """One call doing two things, to avoid a second round-trip per question:

        1. Rewrites the question to be fully self-contained — resolving
           conversational references (e.g. 'its' -> the last discussed
           entity) using prior turns, and positional references (e.g.
           "picture 2", "the second document") using the actual list of
           uploaded documents. Memory and the document list only disambiguate
           the question text itself, never a source of facts (requirement 8).
        2. Classifies whether the question is a broad request for an overview
           of everything uploaded ("give me detailed info", "what's in
           these") rather than a specific, narrow question — most users
           won't name a document when asking broadly, so this can't be a
           keyword check; it needs actual understanding.

        document_names matters even without conversational context: a first
        question like "what's in picture 2?" still needs the document list
        to resolve, and a first question can just as easily be a broad
        overview request. Only skipped when there's nothing to disambiguate
        and too few documents for "broad vs. specific" to be a meaningful
        distinction.
        """
        document_names = document_names or []
        if not memory_context and len(document_names) < 2:
            return QueryUnderstanding(rewritten_query=question, is_broad_overview=False)

        documents_block = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(document_names))

        prompt = (
            "Given the uploaded document list, the conversation history, and a follow-up "
            "question, respond with ONLY a JSON object (no other text) with exactly these keys:\n\n"
            '"rewritten_question": the question rewritten to be fully self-contained. Resolve '
            "pronouns and references using the conversation history. Resolve positional references "
            'to documents (e.g. "picture 2", "the second image", "the second document") using the '
            "numbered document list below, naming the actual filename instead of the position. If "
            "the follow-up question is about a different document or topic than the conversation "
            "history, treat it as a fresh question about that document — do not merge it with the "
            "prior topic.\n\n"
            '"is_broad_overview": true if the question is a broad request for an overview/summary '
            "covering everything uploaded (e.g. \"give me detailed info\", \"what's in these\", "
            '"summarize these"), false if it is a specific question about particular content.\n\n'
            f"Uploaded documents (in upload order):\n{documents_block}\n\n"
            f"Conversation history:\n{memory_context or '(none yet)'}\n\n"
            f"Follow-up question: {question}"
        )
        response = self._client.chat.complete(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
            rewritten = str(parsed.get("rewritten_question") or "").strip()
            return QueryUnderstanding(
                rewritten_query=rewritten or question,
                is_broad_overview=bool(parsed.get("is_broad_overview", False)),
            )
        except (json.JSONDecodeError, AttributeError):
            return QueryUnderstanding(rewritten_query=question, is_broad_overview=False)

    def generate_answer(self, question: str, context_chunks: list[dict], broad_overview: bool = False) -> str:
        """context_chunks must already be relevance-filtered upstream
        (guardrails.grounding.is_relevant) before reaching this call, unless
        broad_overview is set — that path intentionally includes every
        document's chunks unfiltered (see Retriever.retrieve)."""
        if not context_chunks:
            return NOT_FOUND_MESSAGE

        system_prompt = build_grounding_system_prompt(broad_overview=broad_overview)
        context_block = build_context_block(context_chunks)

        user_prompt = f"Document context:\n\n{context_block}\n\nQuestion: {question}"

        response = self._client.chat.complete(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
