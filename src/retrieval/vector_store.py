"""
Vector Embedding and Semantic Index.
Provides dense cosine and lexical similarity search over Knowledge Graph entities
and document chunks to discover seed anchor nodes for multi-hop graph expansion.
"""
from typing import List, Dict, Any, Tuple, Optional
import math
import re
from collections import Counter
from ..parser.schemas import KGEntity


class SimpleDenseVectorStore:
    """
    Self-contained semantic vector store using term frequency-inverse document frequency
    (TF-IDF) and n-gram subword embeddings with cosine similarity.
    Provides fast, zero-dependency semantic vector search over entity descriptions and chunks.
    """

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self._doc_vectors: Dict[str, Dict[str, float]] = {}
        self._idf: Dict[str, float] = {}
        self._vocab: set = set()

    def _tokenize(self, text: str) -> List[str]:
        """Normalize text and generate word + char 3-gram tokens for semantic robustness."""
        clean = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text.lower())
        words = [w for w in clean.split() if len(w) > 1]
        
        # Add subword 3-grams to handle partial matches and terminology variations
        subwords = []
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    subwords.append(f"#{word[i:i+3]}")
        return words + subwords

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a text document or entity description to the vector store."""
        tokens = self._tokenize(text)
        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata or {},
            "tokens": tokens
        }
        self._rebuild_index()

    def add_entity(self, entity: KGEntity) -> None:
        """Index an entity for vector retrieval."""
        full_text = f"{entity.name} {entity.entity_type} {entity.description} " + " ".join(
            f"{k}:{v}" for k, v in entity.properties.items() if isinstance(v, (str, int, float))
        )
        self.add_document(
            doc_id=entity.id,
            text=full_text,
            metadata={"entity": entity.model_dump(), "type": "entity"}
        )

    def _rebuild_index(self):
        """Calculate TF-IDF vector representations across all indexed documents."""
        n_docs = len(self.documents)
        if n_docs == 0:
            return

        # 1. Document Frequencies
        df: Dict[str, int] = Counter()
        for doc in self.documents.values():
            unique_terms = set(doc["tokens"])
            for t in unique_terms:
                df[t] += 1

        # 2. IDF
        self._idf = {
            t: math.log((1 + n_docs) / (1 + count)) + 1.0
            for t, count in df.items()
        }

        # 3. Normalized Doc Vectors
        self._doc_vectors = {}
        for doc_id, doc in self.documents.items():
            tf = Counter(doc["tokens"])
            vec: Dict[str, float] = {}
            norm_sq = 0.0
            for t, count in tf.items():
                w = (1 + math.log(count)) * self._idf.get(t, 1.0)
                vec[t] = w
                norm_sq += w * w
            
            norm = math.sqrt(norm_sq) or 1.0
            self._doc_vectors[doc_id] = {t: val / norm for t, val in vec.items()}

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform cosine similarity vector search for the query."""
        if not self._doc_vectors:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        q_vec: Dict[str, float] = {}
        q_norm_sq = 0.0
        for t, count in q_tf.items():
            idf_val = self._idf.get(t, 1.0)
            w = (1 + math.log(count)) * idf_val
            q_vec[t] = w
            q_norm_sq += w * w
        
        q_norm = math.sqrt(q_norm_sq) or 1.0
        norm_q_vec = {t: val / q_norm for t, val in q_vec.items()}

        scores: List[Tuple[str, float]] = []
        for doc_id, doc_vec in self._doc_vectors.items():
            # Dot product
            score = 0.0
            for t, q_val in norm_q_vec.items():
                if t in doc_vec:
                    score += q_val * doc_vec[t]
            if score > 0.01:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(doc_id, score, self.documents[doc_id]["metadata"]) for doc_id, score in scores[:top_k]]
