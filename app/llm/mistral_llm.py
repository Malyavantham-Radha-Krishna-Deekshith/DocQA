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

    def rewrite_query(self, question: str, memory_context: str) -> str:
        """Resolves conversational references (e.g. 'its' -> the last
        discussed entity) using prior turns. Memory is only used to
        disambiguate the question text itself, never as a source of facts
        (requirement 8)."""
        if not memory_context:
            return question

        prompt = (
            "Given the conversation history and a follow-up question, rewrite the "
            "follow-up question to be fully self-contained by resolving pronouns "
            "and references. Output ONLY the rewritten question, nothing else.\n\n"
            f"Conversation history:\n{memory_context}\n\n"
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
