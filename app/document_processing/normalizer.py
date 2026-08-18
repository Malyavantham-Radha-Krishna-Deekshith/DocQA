"""Normalize raw Mistral OCR markdown before chunking (requirement 3).

Only touches formatting noise (whitespace, stray artifacts). Never rewrites
anything that looks like a critical field (numbers, dates, IDs, phone
numbers, prices, percentages) — those are flagged for the future
critical-field validation layer (requirement 12), not silently "fixed".
"""
import re
from dataclasses import dataclass
from typing import List

# Patterns used to *detect* critical fields so they can be tagged as metadata,
# not to alter them. Deliberately conservative / non-exhaustive for the MVP.
CRITICAL_FIELD_PATTERNS = {
    "ein": re.compile(r"\b\d{2}-\d{7}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "date": re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
    "percentage": re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%"),
    "currency": re.compile(r"[$£€]\s?\d[\d,]*(?:\.\d{2})?"),
    "id_number": re.compile(r"\b[A-Z]{0,3}\d{6,}\b"),
}


@dataclass
class CriticalFieldMatch:
    field_type: str
    value: str
    start: int
    end: int


def extract_critical_fields(text: str) -> List[CriticalFieldMatch]:
    matches: List[CriticalFieldMatch] = []
    for field_type, pattern in CRITICAL_FIELD_PATTERNS.items():
        for m in pattern.finditer(text):
            matches.append(CriticalFieldMatch(field_type, m.group(0), m.start(), m.end()))
    return matches


def _line_has_critical_field(line: str) -> bool:
    return any(p.search(line) for p in CRITICAL_FIELD_PATTERNS.values())


def normalize_markdown(markdown: str) -> str:
    """Collapse OCR whitespace noise line-by-line, skipping lines that
    contain a critical field so we never risk mutating a factual value."""
    lines = markdown.split("\n")
    normalized_lines = []
    for line in lines:
        if _line_has_critical_field(line):
            # Leave critical-field lines untouched except trailing whitespace.
            normalized_lines.append(line.rstrip())
            continue
        # Safe cleanup: collapse repeated spaces, strip trailing whitespace.
        cleaned = re.sub(r"[ \t]{2,}", " ", line).rstrip()
        normalized_lines.append(cleaned)

    # Collapse 3+ blank lines down to a single blank line separator.
    text = "\n".join(normalized_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
