"""
Hybrid Retrieval Engine.
Integrates vector similarity search with graph traversal (k-hop expansion and pathfinding)
to assemble comprehensive multi-hop evidence packages.
"""
from typing import List, Dict, Any, Tuple, Optional
from .vector_store import DenseVectorStore
from ..graph.base import BaseGraphStore
from ..parser.schemas import KGEntity, KGRelation


class HybridRetriever:
    """Combines vector similarity search with knowledge graph traversal."""

    def __init__(self, vector_store: DenseVectorStore, graph_store: BaseGraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def retrieve(
        self,
        query: str,
        top_k_seeds: int = 5,
        k_hops: int = 2,
        find_paths: bool = True,
        use_hipporag: bool = True,
        use_steiner_tree: bool = True
    ) -> Dict[str, Any]:
        """
        Execute advanced hybrid retrieval:
        1. Find seed anchor entities via dense vector similarity.
        2. HippoRAG associative activation via Personalized PageRank (PPR).
        3. Steiner Minimal Tree (SMT) subgraph extraction connecting multi-concept seeds.
        4. K-hop neighborhood expansion with operational edge filtering.
        5. Synthesize noise-pruned structural evidence.
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
                "hipporag_activation": {},
                "formatted_evidence": "No relevant entities found in knowledge base."
            }

        # Step 2: HippoRAG Associative Activation (Personalized PageRank)
        hipporag_scores: Dict[str, float] = {}
        if use_hipporag and hasattr(self.graph_store, "personalized_pagerank"):
            seed_weights = {doc_id: score for doc_id, score, meta in seed_results}
            hipporag_scores = self.graph_store.personalized_pagerank(seed_weights, alpha=0.85)

        # Step 3: Graph expansion (k-hop neighborhood)
        subgraph_entities, subgraph_relations = self.graph_store.get_k_hop_subgraph(
            seed_ids=seed_ids,
            k=k_hops,
            direction="both"
        )

        # Step 4: Steiner Minimal Tree & Bridging Paths
        bridging_paths: List[List[str]] = []
        steiner_entities: List[KGEntity] = []
        steiner_relations: List[KGRelation] = []

        if use_steiner_tree and hasattr(self.graph_store, "extract_steiner_subgraph") and len(seed_ids) >= 2:
            steiner_entities, steiner_relations = self.graph_store.extract_steiner_subgraph(
                terminal_ids=seed_ids[:4],
                operational_only=True
            )
            # Merge Steiner elements into subgraph
            existing_ent_ids = {e.id for e in subgraph_entities}
            for se in steiner_entities:
                if se.id not in existing_ent_ids:
                    subgraph_entities.append(se)
                    existing_ent_ids.add(se.id)

            existing_rel_keys = {(r.source_id, r.target_id, r.relation_type) for r in subgraph_relations}
            for sr in steiner_relations:
                key = (sr.source_id, sr.target_id, sr.relation_type)
                if key not in existing_rel_keys:
                    subgraph_relations.append(sr)
                    existing_rel_keys.add(key)

        if find_paths and len(seed_ids) >= 2:
            for i in range(len(seed_ids)):
                for j in range(i + 1, min(len(seed_ids), 3)):
                    src = seed_ids[i]
                    tgt = seed_ids[j]
                    path = self.graph_store.find_shortest_path(src, tgt)
                    if path and len(path) > 1 and path not in bridging_paths:
                        bridging_paths.append(path)

        # Step 5: Build human-readable formatted evidence
        formatted_evidence = self._format_evidence(
            seed_results=seed_results,
            subgraph_entities=subgraph_entities,
            subgraph_relations=subgraph_relations,
            bridging_paths=bridging_paths,
            hipporag_scores=hipporag_scores
        )

        return {
            "query": query,
            "seed_entities": seed_results,
            "subgraph_entities": subgraph_entities,
            "subgraph_relations": subgraph_relations,
            "bridging_paths": bridging_paths,
            "hipporag_activation": hipporag_scores,
            "formatted_evidence": formatted_evidence
        }

    def _format_evidence(
        self,
        seed_results: List[Tuple[str, float, Dict[str, Any]]],
        subgraph_entities: List[KGEntity],
        subgraph_relations: List[KGRelation],
        bridging_paths: List[List[str]],
        hipporag_scores: Optional[Dict[str, float]] = None
    ) -> str:
        lines: List[str] = []

        lines.append("### 1. Primary Seed Entities (Vector Similarity Match):")
        for doc_id, score, meta in seed_results:
            ent = self.graph_store.get_entity(doc_id)
            ppr_str = ""
            if hipporag_scores and doc_id in hipporag_scores:
                ppr_str = f" | HippoRAG PPR: {hipporag_scores[doc_id]:.4f}"
            if ent:
                lines.append(f"- **{ent.name}** [{ent.entity_type}] (relevance: {score:.3f}{ppr_str}): {ent.description}")
            else:
                lines.append(f"- **{doc_id}** (relevance: {score:.3f}{ppr_str})")

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

