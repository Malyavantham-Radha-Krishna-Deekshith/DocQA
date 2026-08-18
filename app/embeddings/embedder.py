"""Local embedding model wrapper (requirement 5).

Uses sentence-transformers so the MVP has no dependency on a hosted
embedding API. Swapping to a Mistral/OpenAI embedding endpoint later only
means changing this one class.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
