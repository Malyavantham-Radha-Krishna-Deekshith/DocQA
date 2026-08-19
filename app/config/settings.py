"""Central configuration for the Document Q&A RAG pipeline.

Every tunable referenced in the project spec (chunking, overlap presets,
relevance threshold, model names) lives here so modules never hardcode
values individually.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Reads from the environment (.env locally) first, falling back to
    Streamlit Cloud's st.secrets when the env var isn't set there."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# --- API keys ---
MISTRAL_API_KEY = _get_secret("MISTRAL_API_KEY")

# --- Models ---
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
MISTRAL_LLM_MODEL = os.getenv("MISTRAL_LLM_MODEL", "mistral-large-latest")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

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

# --- Session memory ---
MAX_MEMORY_TURNS = 10
