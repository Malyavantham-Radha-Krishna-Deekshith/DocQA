"""Structure-aware chunking (requirement 4).

Parses OCR markdown into typed blocks (heading / paragraph / list / table),
then packs blocks into token-bounded chunks along Document -> Section ->
Heading -> Paragraph -> Table boundaries instead of raw character splitting.

Tables are never blindly overlapped: if a table has to split across chunks,
the header row is repeated in the continuation chunk so each chunk stays
self-contained.
"""
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        # Fallback heuristic if tiktoken/its encoding files aren't available offline.
        return int(len(text.split()) * 1.3)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class Block:
    type: str  # "heading" | "paragraph" | "list" | "table"
    text: str
    level: Optional[int] = None       # heading level, if applicable
    section: str = ""                 # nearest enclosing heading text
    table_header: Optional[str] = None  # for table blocks: the header (+ separator) lines


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    section: str
    text: str
    critical_field_types: List[str] = field(default_factory=list)


def _parse_blocks(markdown: str) -> List[Block]:
    """Split page markdown into typed blocks, tracking the current section heading."""
    lines = markdown.split("\n")
    blocks: List[Block] = []
    current_section = ""
    buffer: List[str] = []
    buffer_type = None  # "paragraph" | "list"

    def flush_buffer():
        nonlocal buffer, buffer_type
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(Block(type=buffer_type or "paragraph", text=text, section=current_section))
        buffer = []
        buffer_type = None

    i = 0
    while i < len(lines):
        line = lines[i]

        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_buffer()
            level, title = len(heading_match.group(1)), heading_match.group(2).strip()
            current_section = title
            blocks.append(Block(type="heading", text=title, level=level, section=current_section))
            i += 1
            continue

        if TABLE_ROW_RE.match(line):
            flush_buffer()
            table_lines = []
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            header = "\n".join(table_lines[:2]) if len(table_lines) >= 2 else table_lines[0]
            blocks.append(Block(
                type="table",
                text="\n".join(table_lines),
                section=current_section,
                table_header=header,
            ))
            continue

        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "1. ")) or re.match(r"^\d+\.\s", stripped):
            if buffer_type not in (None, "list"):
                flush_buffer()
            buffer_type = "list"
            buffer.append(line)
            i += 1
            continue

        if stripped == "":
            flush_buffer()
            i += 1
            continue

        if buffer_type not in (None, "paragraph"):
            flush_buffer()
        buffer_type = "paragraph"
        buffer.append(line)
        i += 1

    flush_buffer()
    return blocks


def _split_table_block(block: Block, max_tokens: int) -> List[str]:
    """Split an oversized table into row groups, repeating the header in each piece."""
    lines = block.text.split("\n")
    header_lines = block.table_header.split("\n") if block.table_header else lines[:2]
    data_rows = lines[len(header_lines):]

    pieces: List[str] = []
    current_rows: List[str] = []
    header_tokens = count_tokens("\n".join(header_lines))

    def flush():
        if current_rows:
            pieces.append("\n".join(header_lines + current_rows))

    running = header_tokens
    for row in data_rows:
        row_tokens = count_tokens(row)
        if running + row_tokens > max_tokens and current_rows:
            flush()
            current_rows = []
            running = header_tokens
        current_rows.append(row)
        running += row_tokens
    flush()
    return pieces or [block.text]


def _pack_blocks(
    blocks: List[Block],
    max_tokens: int,
    overlap_ratio: float,
) -> List[List[Block]]:
    """Greedily pack blocks into chunks under max_tokens, carrying forward the
    last overlap_ratio fraction of the previous chunk's blocks for continuity.
    Tables are handled separately (split with repeated headers) rather than
    being subject to blind text overlap."""
    groups: List[List[Block]] = []
    current: List[Block] = []
    current_tokens = 0

    def start_new_group_with_overlap():
        nonlocal current, current_tokens
        if not current or overlap_ratio <= 0:
            current, current_tokens = [], 0
            return
        target_overlap_tokens = int(max_tokens * overlap_ratio)
        overlap_blocks: List[Block] = []
        running = 0
        for b in reversed(current):
            if b.type == "table":
                break  # never carry a table into overlap
            t = count_tokens(b.text)
            if running + t > target_overlap_tokens:
                break
            overlap_blocks.insert(0, b)
            running += t
        current, current_tokens = overlap_blocks, running

    for block in blocks:
        block_tokens = count_tokens(block.text)

        if block.type == "table" and block_tokens > max_tokens:
            if current:
                groups.append(current)
                start_new_group_with_overlap()
            for piece_text in _split_table_block(block, max_tokens):
                groups.append([Block(type="table", text=piece_text, section=block.section)])
            continue

        if current_tokens + block_tokens > max_tokens and current:
            groups.append(current)
            start_new_group_with_overlap()

        current.append(block)
        current_tokens += block_tokens

    if current:
        groups.append(current)

    return groups


def chunk_document(
    ocr_result,
    max_tokens: int = 650,
    overlap_ratio: float = 0.12,
) -> List[Chunk]:
    """ocr_result: OCRDocumentResult (already normalized markdown per page)."""
    from app.document_processing.normalizer import extract_critical_fields

    all_chunks: List[Chunk] = []
    for page in ocr_result.pages:
        blocks = _parse_blocks(page.markdown)
        groups = _pack_blocks(blocks, max_tokens, overlap_ratio)

        for group in groups:
            text = "\n\n".join(b.text for b in group)
            if not text.strip():
                continue
            section = next((b.section for b in group if b.section), "")
            critical = extract_critical_fields(text)
            all_chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=ocr_result.document_id,
                filename=ocr_result.filename,
                page=page.page_number,
                section=section,
                text=text,
                critical_field_types=sorted({m.field_type for m in critical}),
            ))

    return all_chunks
