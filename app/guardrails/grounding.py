"""Anti-hallucination and relevance guardrails (requirements 9, 10, 11).

These are the rules the rest of the pipeline is built around: don't send
irrelevant context to the LLM, and don't let the LLM answer from anything
but retrieved document content.
"""
from typing import List

from app.config import settings

NOT_FOUND_MESSAGE = settings.NOT_FOUND_MESSAGE


def is_relevant(top_score: float, threshold: float = settings.RELEVANCE_THRESHOLD) -> bool:
    """Gate: only chunks meeting this similarity threshold are allowed to
    reach the LLM. Below threshold -> refuse rather than answer from
    general knowledge (requirement 10)."""
    return top_score >= threshold


def build_grounding_system_prompt(broad_overview: bool = False) -> str:
    prompt = f"""You are a document question-answering assistant. Follow these rules strictly:

1. Answer ONLY using the provided document context below. Do not use any external or general knowledge.
2. Do not guess or infer information that is not explicitly present in the context.
3. Do not fabricate names, numbers, dates, prices, IDs, percentages, or any other facts.
4. Preserve numeric values EXACTLY as they appear in the source context — do not round, reformat, or "correct" them.
5. If the context does not contain the answer, respond exactly with: "{NOT_FOUND_MESSAGE}"
6. If different chunks contain conflicting information, explicitly state the conflict and cite both sources instead of picking one.
7. The UI already shows source citations separately below your answer — do not restate "Source: ..." or list sources at the end of your answer.
8. Use no Markdown syntax (no **bold**, no ##, no numbered-list periods) — the answer is rendered as plain text, so it would show up literally. Instead, format for readability with plain text alone:
   - A short, single-fact answer: one plain sentence, nothing else.
   - Several distinct facts: one per line, each starting with "- ".
   - Facts that fall into distinct groups (e.g. different sections of a form): a short label for the group on its own line, followed by its "- " bullet lines, with a blank line between groups.
"""
    if broad_overview:
        prompt += (
            "9. This is a broad overview request covering everything uploaded, not one specific fact — "
            "the context below includes every uploaded document, not just ones matched to the question. "
            "Organize your answer by document: one section per document, using its filename as the "
            "section label, followed by that document's key facts as \"- \" bullet lines. If a "
            "particular document has nothing relevant to contribute, skip that document's section "
            "entirely rather than forcing something. Only fall back to rule 5's not-found message if "
            "NONE of the documents have anything relevant to contribute.\n"
        )
    return prompt


def format_sources(chunks: List[dict]) -> str:
    """chunks: list of metadata dicts as returned by FaissStore.search()."""
    lines = []
    seen = set()
    for c in chunks:
        key = (c["filename"], c.get("page"), c.get("section"))
        if key in seen:
            continue
        seen.add(key)
        section = f" | Section: {c['section']}" if c.get("section") else ""
        lines.append(f"- {c['filename']} (page {c.get('page', '?')}){section}")
    return "\n".join(lines)


def build_context_block(chunks: List[dict]) -> str:
    """Assembles retrieved chunks into the context block handed to the LLM,
    each one tagged with its source so the model can cite it."""
    parts = []
    for c in chunks:
        header = f"[Source: {c['filename']}, page {c.get('page', '?')}, section: {c.get('section') or 'N/A'}]"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)
