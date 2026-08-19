"""Central configuration for the Document Q&A RAG pipeline.

Every tunable referenced in the project spec (chunking, overlap presets,
relevance threshold, model names) lives here so modules never hardcode
values individually.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# --- API keys ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# --- Models ---
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
MISTRAL_LLM_MODEL = os.getenv("MISTRAL_LLM_MODEL", "mistral-large-latest")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "mistral-embed")

# --- Upload constraints ---
MIN_IMAGES = 1
MAX_IMAGES = 20
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]

# --- Chunking ---
# Spec: 500-800 tokens, ~10-15% overlap by default, configurable for benchmarking.
CHUNK_SIZE_TOKENS = 650
DEFAULT_OVERLAP_RATIO = 0.12

# Named presets for later retrieval/answer-accuracy benchmarking (requirement 4).
OVERLAP_PRESETS = {
    "0%": 0.0,
    "10%": 0.10,
    "20%": 0.20,
}

# --- Retrieval / guardrails ---
TOP_K = 5
# Cosine similarity threshold (embeddings are L2-normalized before indexing).
# Placeholder — tune once we have real retrieval eval data (requirement 15).
RELEVANCE_THRESHOLD = 0.35

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded documents."

# Cap on how many chunks a broad "overview of everything" question can send
# to the LLM in one call, protecting the context window / cost when a
# session has a large number of indexed documents.
MAX_BROAD_CONTEXT_CHUNKS = 60

# --- Session memory ---
MAX_MEMORY_TURNS = 10

# --- API session lifecycle ---
# How long a session's in-memory documents/chat history survive with no
# activity before being evicted, and how often the sweep runs.
SESSION_IDLE_SECONDS = 30 * 60
SESSION_SWEEP_INTERVAL_SECONDS = 60

# --- CORS ---
# Comma-separated list of allowed frontend origins (the deployed Vercel URL
# plus local dev servers). Set CORS_ORIGINS in the environment for prod.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
