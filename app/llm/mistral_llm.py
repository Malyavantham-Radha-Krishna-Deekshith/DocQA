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
    is_conversational: bool = False
    conversational_reply: str = ""


class MistralLLMClient:
    def __init__(self, api_key: str = settings.MISTRAL_API_KEY, model: str = settings.MISTRAL_LLM_MODEL):
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self._client = Mistral(api_key=api_key)
        self._model = model

    def understand_query(
        self, question: str, memory_context: str, document_names: list[str] | None = None
    ) -> QueryUnderstanding:
        """One call doing three things, to avoid extra round-trips per message:

        1. Classifies whether this is conversational small talk (a greeting,
           thanks, "how are you", etc.) rather than an actual question about
           the uploaded documents — and if so, writes a short, warm reply
           directly, as a friendly document Q&A assistant. This runs on
           every message, including the very first one with nothing
           uploaded yet, since a greeting is just as likely to open a
           conversation as follow it.
        2. Rewrites real questions to be fully self-contained — resolving
           conversational references (e.g. 'its' -> the last discussed
           entity) using prior turns, and positional references (e.g.
           "picture 2", "the second document") using the actual list of
           uploaded documents. Memory and the document list only disambiguate
           the question text itself, never a source of facts (requirement 8).
        3. Classifies whether a real question is a broad request for an
           overview of everything uploaded ("give me detailed info", "what's
           in these") rather than a specific, narrow question — most users
           won't name a document when asking broadly, so this can't be a
           keyword check; it needs actual understanding.
        """
        document_names = document_names or []
        documents_block = (
            "\n".join(f"{i + 1}. {name}" for i, name in enumerate(document_names))
            if document_names
            else "(none uploaded yet)"
        )

        prompt = (
            "Given the uploaded document list, the conversation history, and a follow-up "
            "message, respond with ONLY a JSON object (no other text) with exactly these keys:\n\n"
            '"is_conversational": true if the message is a greeting, casual small talk, or '
            'pleasantry (e.g. "hi", "hai", "how are you", "thanks") rather than an actual question '
            "about the uploaded documents. false otherwise.\n\n"
            '"conversational_reply": only meaningful when is_conversational is true — a short, warm, '
            "natural reply (1-2 sentences), as a friendly assistant for a document Q&A app. If no "
            "documents are uploaded yet, gently invite them to upload one; otherwise you may "
            "acknowledge the conversation so far. Never answer factual questions from general "
            'knowledge here. Empty string if is_conversational is false.\n\n'
            '"rewritten_question": only meaningful when is_conversational is false — the message '
            "rewritten to be fully self-contained. Resolve pronouns and references using the "
            'conversation history. Resolve positional references to documents (e.g. "picture 2", '
            '"the second image", "the second document") using the numbered document list below, '
            "naming the actual filename instead of the position. If the message is about a "
            "different document or topic than the conversation history, treat it as a fresh "
            "question about that document — do not merge it with the prior topic.\n\n"
            '"is_broad_overview": only meaningful when is_conversational is false — true if the '
            "question is a broad request for an overview/summary covering everything uploaded "
            '(e.g. "give me detailed info", "what\'s in these", "summarize these"), false if it is '
            "a specific question about particular content.\n\n"
            f"Uploaded documents (in upload order):\n{documents_block}\n\n"
            f"Conversation history:\n{memory_context or '(none yet)'}\n\n"
            f"Follow-up message: {question}"
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
            if bool(parsed.get("is_conversational", False)):
                reply = str(parsed.get("conversational_reply") or "").strip()
                return QueryUnderstanding(
                    rewritten_query=question,
                    is_broad_overview=False,
                    is_conversational=True,
                    conversational_reply=reply or "Hi! Upload a document and ask me anything about it.",
                )
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
