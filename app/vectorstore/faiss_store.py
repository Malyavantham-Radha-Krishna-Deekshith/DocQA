"""FAISS-backed vector store, kept strictly separate from conversation
memory (requirement 6). Stores chunk metadata alongside vectors so every
retrieval result can be traced back to document_id / filename / page /
section / chunk_id (requirement 11).
"""
import json
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from app.chunking.structure_chunker import Chunk


class FaissStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)  # cosine sim via normalized vectors
        self._metadata: List[dict] = []  # parallel to index rows

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        assert len(chunks) == embeddings.shape[0]
        self._index.add(embeddings)
        for chunk in chunks:
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "section": chunk.section,
                "text": chunk.text,
                "critical_field_types": chunk.critical_field_types,
            })

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[dict]:
        """Returns metadata dicts (each with an added 'score' key), best first.
        If document_id is set, filters to that document (metadata filtering,
        requirement 6) by over-fetching then filtering, since FAISS's flat
        index has no native filter support."""
        if self._index.ntotal == 0:
            return []

        fetch_k = top_k if document_id is None else min(self._index.ntotal, top_k * 5)
        scores, indices = self._index.search(query_embedding.reshape(1, -1), fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = dict(self._metadata[idx])
            if document_id is not None and meta["document_id"] != document_id:
                continue
            meta["score"] = float(score)
            results.append(meta)
            if len(results) >= top_k:
                break
        return results

    @property
    def is_empty(self) -> bool:
        return self._index.ntotal == 0

    @property
    def document_filenames(self) -> List[str]:
        """Unique filenames in upload order, for resolving positional
        references like "picture 2" during query rewriting."""
        seen: dict[str, None] = {}
        for meta in self._metadata:
            seen.setdefault(meta["filename"], None)
        return list(seen.keys())

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_path / "index.faiss"))
        with open(dir_path / "metadata.pkl", "wb") as f:
            pickle.dump(self._metadata, f)

    @classmethod
    def load(cls, dir_path: Path, dimension: int) -> "FaissStore":
        store = cls(dimension)
        index_path = dir_path / "index.faiss"
        meta_path = dir_path / "metadata.pkl"
        if index_path.exists() and meta_path.exists():
            store._index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                store._metadata = pickle.load(f)
        return store
