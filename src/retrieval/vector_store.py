"""
Dense Vector Embedding Store using FastEmbed (ONNX Runtime).
Produces authentic 384-dimensional dense embeddings (BAAI/bge-small-en-v1.5)
and computes exact cosine similarity for seed entity and chunk retrieval.
Includes lightweight lexical fallback for zero-dependency offline environments.
"""
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
from ..parser.schemas import KGEntity

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


class DenseVectorStore:
    """
    Production-grade dense vector store.
    Uses ONNX-quantized Transformer embeddings (BAAI/bge-small-en-v1.5)
    to perform true dense semantic search over knowledge graph entities and text chunks.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, 384)
        
        if FASTEMBED_AVAILABLE:
            try:
                self._model = TextEmbedding(model_name=model_name)
            except Exception:
                self._model = None

    @property
    def is_dense(self) -> bool:
        return self._model is not None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Compute normalized dense embeddings for a batch of strings."""
        if self._model is not None:
            # fastembed yields normalized numpy arrays
            embs = list(self._model.embed(texts))
            arr = np.array(embs, dtype=np.float32)
            # Ensure L2 normalization
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            return arr / norms
        else:
            # Fallback pseudo-dense hash projection if model unavailable
            return self._fallback_hash_embed(texts)

    def _fallback_hash_embed(self, texts: List[str], dim: int = 128) -> np.ndarray:
        """Deterministic hash-based dense projection fallback."""
        vectors = []
        for t in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for w in t.lower().split():
                h = abs(hash(w)) % dim
                vec[h] += 1.0
            norm = np.linalg.norm(vec) or 1.0
            vectors.append(vec / norm)
        return np.array(vectors, dtype=np.float32)

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a document or chunk to the index."""
        emb = self.embed_texts([text])  # shape (1, D)
        self.doc_ids.append(doc_id)
        self.doc_texts.append(text)
        self.doc_metadata.append(metadata or {})

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

    def add_entity(self, entity: KGEntity) -> None:
        """Index a Knowledge Graph entity for dense semantic retrieval."""
        props = " ".join(f"{k}: {v}" for k, v in entity.properties.items() if isinstance(v, (str, int, float)))
        full_text = f"Entity: {entity.name}. Type: {entity.entity_type}. Description: {entity.description}. Details: {props}"
        self.add_document(
            doc_id=entity.id,
            text=full_text,
            metadata={"entity": entity.model_dump(), "type": "entity"}
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform dense cosine similarity search."""
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []

        q_emb = self.embed_texts([query])[0]  # shape (D,)
        
        # Cosine similarity via inner product on normalized vectors
        scores = np.dot(self.embeddings, q_emb)  # shape (N,)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.15:  # Relevance cutoff
                results.append((self.doc_ids[idx], score, self.doc_metadata[idx]))
        
        return results

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic cosine similarity between two arbitrary texts."""
        embs = self.embed_texts([text_a, text_b])
        return float(np.dot(embs[0], embs[1]))


# Backwards compatibility alias
SimpleDenseVectorStore = DenseVectorStore
