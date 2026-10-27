"""
Standard Vector RAG Implementation.
Simulates traditional chunk-based vector retrieval (naive text chunking + dense vector search)
to provide a rigorous, empirical baseline for multi-hop evaluation against GraphRAG.
"""
from typing import List, Dict, Any, Tuple
import re
from .vector_store import SimpleDenseVectorStore


class StandardVectorRAG:
    """
    Simulates standard, chunk-based RAG architecture:
    1. Chunks raw text & markdown into sliding windows without graph topology.
    2. Performs top-k semantic search.
    3. Concatenates isolated snippets for LLM synthesis.
    """

    def __init__(self, chunk_size: int = 180, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = SimpleDenseVectorStore()
        self.chunks: List[Dict[str, Any]] = []

    def ingest_raw_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Chunk and index raw unstructured texts, tables, and architecture descriptions.
        documents: List of dicts with 'doc_id', 'title', 'content'.
        """
        chunk_count = 0
        for doc in documents:
            doc_id = doc.get("doc_id", "doc")
            title = doc.get("title", "")
            content = doc.get("content", "")
            
            # Simple word-based sliding chunker
            words = content.split()
            stride = max(1, self.chunk_size - self.chunk_overlap)
            
            for i in range(0, len(words), stride):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                cid = f"{doc_id}_chunk_{i}"
                
                self.chunks.append({
                    "chunk_id": cid,
                    "doc_title": title,
                    "text": chunk_text
                })
                self.vector_store.add_document(
                    doc_id=cid,
                    text=f"{title}\n{chunk_text}",
                    metadata={"doc_title": title, "text": chunk_text}
                )
                chunk_count += 1

        return chunk_count

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Execute standard vector retrieval and assemble naive snippet context."""
        matches = self.vector_store.search(query, top_k=top_k)
        
        retrieved_snippets = []
        for cid, score, meta in matches:
            retrieved_snippets.append({
                "chunk_id": cid,
                "score": score,
                "text": meta.get("text", "")
            })

        context_text = "\n---\n".join(s["text"] for s in retrieved_snippets)
        return {
            "query": query,
            "retrieved_snippets": retrieved_snippets,
            "assembled_context": context_text
        }
