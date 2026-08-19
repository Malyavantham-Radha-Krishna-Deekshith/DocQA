"""Mistral-hosted embedding wrapper (requirement 5).

Calls Mistral's embeddings API rather than running a local
sentence-transformers model — the local model (torch + transformers)
needs far more RAM than fits in a constrained deployment (Render's free
tier is 512MB; that alone OOM-killed the process), and this app already
holds a Mistral API key for OCR/chat, so no new credential is needed.
"""
import numpy as np

from mistralai.client import Mistral

from app.config import settings

# mistral-embed always returns 1024-dim vectors; no local model to
# introspect for this the way sentence-transformers offered.
EMBEDDING_DIMENSION = 1024


class Embedder:
    def __init__(self, api_key: str = settings.MISTRAL_API_KEY, model_name: str = settings.EMBEDDING_MODEL_NAME):
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self._client = Mistral(api_key=api_key)
        self._model_name = model_name
        self.dimension = EMBEDDING_DIMENSION

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model_name, inputs=texts)
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
