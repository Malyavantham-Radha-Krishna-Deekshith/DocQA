"""In-process session memory (requirement 8).

Deliberately NOT a source of facts — it only carries conversational
context (prior question/answer turns) so the query-understanding step can
resolve references like "its" -> "ACME Corp Software". All actual facts
still have to come from document retrieval.
"""
from dataclasses import dataclass, field
from typing import List

from app.config import settings


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class SessionMemory:
    turns: List[Turn] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))
        if len(self.turns) > settings.MAX_MEMORY_TURNS:
            self.turns.pop(0)

    def as_context_string(self) -> str:
        """Rendered history for the query-rewriting LLM call. Empty string
        if this is the first turn."""
        if not self.turns:
            return ""
        lines = []
        for t in self.turns:
            lines.append(f"User: {t.question}")
            lines.append(f"Assistant: {t.answer}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns.clear()
