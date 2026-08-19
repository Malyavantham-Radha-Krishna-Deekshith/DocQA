"""Mistral chat client for query rewriting and grounded answer generation."""
from mistralai.client import Mistral

from app.config import settings
from app.guardrails.grounding import build_grounding_system_prompt, build_context_block, NOT_FOUND_MESSAGE


class MistralLLMClient:
    def __init__(self, api_key: str = settings.MISTRAL_API_KEY, model: str = settings.MISTRAL_LLM_MODEL):
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self._client = Mistral(api_key=api_key)
        self._model = model

    def rewrite_query(self, question: str, memory_context: str, document_names: list[str] | None = None) -> str:
        """Resolves conversational references (e.g. 'its' -> the last
        discussed entity) using prior turns, and positional references (e.g.
        "picture 2", "the second document") using the actual list of
        uploaded documents. Memory and the document list are only used to
        disambiguate the question text itself, never as a source of facts
        (requirement 8).

        document_names matters even without conversational context: a first
        question like "what's in picture 2?" still needs the document list
        to resolve, so rewriting isn't gated on memory_context alone.
        """
        document_names = document_names or []
        if not memory_context and len(document_names) < 2:
            return question

        documents_block = (
            "\n".join(f"{i + 1}. {name}" for i, name in enumerate(document_names))
            if document_names
            else "(none uploaded yet)"
        )

        prompt = (
            "Given the uploaded document list, the conversation history, and a follow-up "
            "question, rewrite the follow-up question to be fully self-contained. "
            "Resolve pronouns and references using the conversation history. Resolve "
            "positional references to documents (e.g. \"picture 2\", \"the second image\", "
            "\"the second document\") using the numbered document list below, naming the "
            "actual filename instead of the position.\n\n"
            "If the follow-up question is about a different document or topic than the "
            "conversation history, treat it as a fresh question about that document — do "
            "not merge it with the prior topic. Output ONLY the rewritten question, "
            "nothing else.\n\n"
            f"Uploaded documents (in upload order):\n{documents_block}\n\n"
            f"Conversation history:\n{memory_context or '(none yet)'}\n\n"
            f"Follow-up question: {question}\n\n"
            "Rewritten standalone question:"
        )
        response = self._client.chat.complete(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten or question

    def generate_answer(self, question: str, context_chunks: list[dict]) -> str:
        """context_chunks must already be relevance-filtered upstream
        (guardrails.grounding.is_relevant) before reaching this call."""
        if not context_chunks:
            return NOT_FOUND_MESSAGE

        system_prompt = build_grounding_system_prompt()
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
