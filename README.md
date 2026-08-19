# Document Q&A RAG

End-to-end image-based Document Q&A app: upload or photograph documents,
extract structured content with Mistral OCR, index it in FAISS, and answer
questions with strict grounding / anti-hallucination guardrails.

A FastAPI backend (deployed on Render) serves a vanilla JS + Tailwind
frontend (deployed on Vercel). Each visitor gets an isolated, in-memory
session — their own documents and chat history, kept separate from every
other visitor.

## Backend setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env      # then fill in MISTRAL_API_KEY
```

## Run the backend

```bash
uvicorn app.main:app --reload
```

Serves on `http://localhost:8000`. Check `http://localhost:8000/api/health`.

## Run the frontend

```bash
cd frontend
npm install
npm run build       # compiles Tailwind to dist/output.css
```

Then serve `frontend/` with any static server (e.g. `npx serve frontend`,
or the VS Code Live Server extension) and open it in a browser.
`frontend/src/config.js` auto-detects `localhost` and points at
`http://localhost:8000`; edit `RENDER_API_URL` in that file to your deployed
backend URL before deploying the frontend.

## Deploying

- **Backend (Render)**: connect the repo, start command
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, set `MISTRAL_API_KEY`
  as an environment variable, and set `CORS_ORIGINS` to your Vercel URL
  (comma-separated if you need more than one origin).
- **Frontend (Vercel)**: set the project root to `frontend/`; Vercel picks up
  `frontend/vercel.json` (`npm install` + `npm run build`) automatically.

Render's free tier spins the backend down after ~15 minutes idle; the first
request after that can take up to ~50s to wake it back up. The frontend
handles this with a "waking up the server…" loading state instead of
appearing frozen.

## Architecture

```
app/
  config/               # settings: chunk size/overlap presets, thresholds, model/session/CORS config
  ocr/                  # Mistral OCR wrapper -> structured markdown + metadata
  document_processing/  # normalization, critical-field detection
  chunking/              # structure-aware chunker (heading/section/table-aware, configurable overlap)
  embeddings/           # local sentence-transformers wrapper
  vectorstore/          # FAISS store + metadata (separate from conversation memory)
  memory/               # session memory (conversational context only, not a fact source)
  retrieval/            # query rewriting -> embed -> FAISS search -> relevance threshold
  llm/                  # Mistral chat client (query rewriting + grounded answer generation)
  guardrails/           # grounding system prompt, relevance gate, source formatting
  evaluation/           # OCR-level and RAG-level metric interfaces (to be filled in)
  session_manager.py    # per-session FaissStore + SessionMemory, shared OCR/embedder/LLM clients
  pipeline.py            # orchestrates OCR -> chunk -> embed -> retrieve -> answer; UI-agnostic
  main.py                 # FastAPI app: /api/health, /api/documents, /api/chat

frontend/
  index.html             # single-page UI: add documents -> preview -> process -> chat
  src/main.js             # DOM wiring, drag & drop, camera capture, chat rendering
  src/api.js              # fetch wrapper, session id, Render wake-up polling
  src/config.js           # backend base URL
  src/input.css           # Tailwind entry point
```

Chunking overlap presets (0% / 10% / 20%) live in `app/config/settings.py`
(`OVERLAP_PRESETS`) for later retrieval/answer-accuracy benchmarking.

## Scope

Sessions are in-memory only (no S3/Redis/Postgres) — a Render restart clears
everyone's documents and chat history, which is fine given Render's disk is
ephemeral anyway. Idle sessions (30+ minutes with no activity) are swept
automatically to bound memory growth.
