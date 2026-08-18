# Document Q&A RAG

Local, end-to-end image-based Document Q&A app: upload or photograph
documents, extract structured content with Mistral OCR, index it in FAISS,
and answer questions with strict grounding / anti-hallucination guardrails.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env      # then fill in MISTRAL_API_KEY
```

## Run

```bash
streamlit run app/ui/streamlit_app.py
```

## Architecture

```
app/
  config/             # settings: chunk size/overlap presets, thresholds, model names
  ocr/                # Mistral OCR wrapper -> structured markdown + metadata
  document_processing/  # normalization, critical-field detection
  chunking/           # structure-aware chunker (heading/section/table-aware, configurable overlap)
  embeddings/         # local sentence-transformers wrapper
  vectorstore/        # FAISS store + metadata (separate from conversation memory)
  memory/             # session memory (conversational context only, not a fact source)
  retrieval/          # query rewriting -> embed -> FAISS search -> relevance threshold
  llm/                # Mistral chat client (query rewriting + grounded answer generation)
  guardrails/         # grounding system prompt, relevance gate, source formatting
  evaluation/         # OCR-level and RAG-level metric interfaces (to be filled in)
  ui/                 # Streamlit app
  pipeline.py         # orchestrates the full flow; UI-agnostic
```

Chunking overlap presets (0% / 10% / 20%) live in `app/config/settings.py`
(`OVERLAP_PRESETS`) for later retrieval/answer-accuracy benchmarking.

## Scope

Local MVP only: no S3, Redis, Postgres, or multi-agent orchestration yet.
Modules are separated so a FastAPI + cloud deployment can be layered on
later without changing the RAG pipeline itself.
