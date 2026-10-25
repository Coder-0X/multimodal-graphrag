"""
Hybrid Retrieval Engine.
Integrates vector similarity search with graph traversal (k-hop expansion and pathfinding)
to assemble comprehensive multi-hop evidence packages.
"""
from typing import List, Dict, Any, Tuple, Optional
from .vector_store import SimpleDenseVectorStore
from ..graph.base import BaseGraphStore
from ..parser.schemas import KGEntity, KGRelation


class HybridRetriever:
    """Combines vector similarity search with knowledge graph traversal."""

    def __init__(self, vector_store: SimpleDenseVectorStore, graph_store: BaseGraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def retrieve(
        self,
        query: str,
        top_k_seeds: int = 4,
        k_hops: int = 1,
        find_paths: bool = True
    ) -> Dict[str, Any]:
        """
        Execute hybrid retrieval:
        1. Find seed anchor entities via vector similarity.
        2. Expand k-hop subgraph around seed entities.
        3. If multiple seeds exist, discover shortest paths connecting them.
        4. Synthesize structural evidence.
        """
        # Step 1: Vector similarity for seeds
        seed_results = self.vector_store.search(query, top_k=top_k_seeds)
        seed_ids = [doc_id for doc_id, score, meta in seed_results]

        if not seed_ids:
            return {
                "query": query,
                "seed_entities": [],
                "subgraph_entities": [],
                "subgraph_relations": [],
                "bridging_paths": [],
                "formatted_evidence": "No relevant entities found in knowledge base."
            }

        # Step 2: Graph expansion (k-hop neighborhood)
        subgraph_entities, subgraph_relations = self.graph_store.get_k_hop_subgraph(
            seed_ids=seed_ids,
            k=k_hops,
            direction="both"
        )

        # Step 3: Bridging paths between seeds
        bridging_paths: List[List[str]] = []
        if find_paths and len(seed_ids) >= 2:
            for i in range(len(seed_ids)):
                for j in range(i + 1, min(len(seed_ids), 3)):
                    src = seed_ids[i]
                    tgt = seed_ids[j]
                    path = self.graph_store.find_shortest_path(src, tgt)
                    if path and len(path) > 1:
                        bridging_paths.append(path)

        # Step 4: Build human-readable formatted evidence
        formatted_evidence = self._format_evidence(
            seed_results=seed_results,
            subgraph_entities=subgraph_entities,
            subgraph_relations=subgraph_relations,
            bridging_paths=bridging_paths
        )

        return {
            "query": query,
            "seed_entities": seed_results,
            "subgraph_entities": subgraph_entities,
            "subgraph_relations": subgraph_relations,
            "bridging_paths": bridging_paths,
            "formatted_evidence": formatted_evidence
        }

    def _format_evidence(
        self,
        seed_results: List[Tuple[str, float, Dict[str, Any]]],
        subgraph_entities: List[KGEntity],
        subgraph_relations: List[KGRelation],
        bridging_paths: List[List[str]]
    ) -> str:
        lines: List[str] = []

        lines.append("### 1. Primary Seed Entities (Vector Similarity Match):")
        for doc_id, score, meta in seed_results:
            ent = self.graph_store.get_entity(doc_id)
            if ent:
                lines.append(f"- **{ent.name}** [{ent.entity_type}] (relevance: {score:.3f}): {ent.description}")
            else:
                lines.append(f"- **{doc_id}** (relevance: {score:.3f})")

        if bridging_paths:
            lines.append("\n### 2. Multi-Hop Relational Paths (Bridging Concepts):")
            for path in bridging_paths:
                named_path = []
                for nid in path:
                    node_ent = self.graph_store.get_entity(nid)
                    named_path.append(node_ent.name if node_ent else nid)
                lines.append(f"- {' --> '.join(named_path)}")

        lines.append(f"\n### 3. Knowledge Graph Subgraph Triples ({len(subgraph_relations)} relations):")
        for rel in subgraph_relations[:25]:  # limit for prompt budget
            src_ent = self.graph_store.get_entity(rel.source_id)
            tgt_ent = self.graph_store.get_entity(rel.target_id)
            src_name = src_ent.name if src_ent else rel.source_id
            tgt_name = tgt_ent.name if tgt_ent else rel.target_id
            desc = f" ({rel.description})" if rel.description else ""
            lines.append(f"- ({src_name}) --[{rel.relation_type}]--> ({tgt_name}){desc}")

        lines.append(f"\n### 4. Component & Metric Specifications ({len(subgraph_entities)} entities):")
        for ent in subgraph_entities[:15]:
            props_str = ", ".join(f"{k}={v}" for k, v in ent.properties.items() if k not in ("footnotes",))
            lines.append(f"- **{ent.name}** ({ent.entity_type}): {ent.description} | [{props_str}]")

        return "\n".join(lines)
