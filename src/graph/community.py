"""
Hierarchical Community Summarization & Global Search Engine.
Implements Microsoft GraphRAG-inspired modular community detection,
cluster-level synthesis, and global cross-cutting risk analysis.
"""
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
from .base import BaseGraphStore
from ..parser.schemas import KGEntity, KGRelation


class CommunitySummary:
    def __init__(self, community_id: int, title: str, entities: List[KGEntity], relations: List[KGRelation]):
        self.community_id = community_id
        self.title = title
        self.entities = entities
        self.relations = relations
        self.entity_ids = [e.id for e in entities]
        self.summary_text = ""
        self.risk_assessment = ""

    def generate_summary(self) -> str:
        """Synthesize a structured thematic summary for this graph community."""
        types_count = defaultdict(int)
        for e in self.entities:
            types_count[e.entity_type] += 1
        
        type_str = ", ".join(f"{cnt} {t}" for t, cnt in types_count.items())
        
        lines = [
            f"**Community {self.community_id}: {self.title}** ({type_str})",
            f"• Scope: Encompasses {len(self.entities)} entities and {len(self.relations)} internal dependencies."
        ]

        # Key entities
        key_names = [e.name for e in self.entities[:5]]
        lines.append(f"• Key Entities: {', '.join(key_names)}")

        # Relational highlights
        if self.relations:
            rel_highlights = []
            for r in self.relations[:3]:
                rel_highlights.append(f"({r.source_id}) --[{r.relation_type}]--> ({r.target_id})")
            lines.append("• Critical Pathways: " + "; ".join(rel_highlights))

        self.summary_text = "\n".join(lines)
        return self.summary_text


class HierarchicalCommunityEngine:
    """Manages community detection, summary generation, and global search routing."""

    def __init__(self, graph_store: BaseGraphStore, llm_client: Any = None):
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.communities: Dict[int, CommunitySummary] = {}
        self.is_built = False

    def build_communities(self) -> Dict[int, CommunitySummary]:
        """Detect modular graph communities and generate hierarchical cluster summaries."""
        raw_clusters = self.graph_store.detect_communities()
        self.communities.clear()

        for comm_id, node_ids in raw_clusters.items():
            if not node_ids:
                continue

            ents: List[KGEntity] = []
            for nid in node_ids:
                e = self.graph_store.get_entity(nid)
                if e:
                    ents.append(e)

            # Extract internal relations
            sub_ents, sub_rels = self.graph_store.get_k_hop_subgraph(seed_ids=node_ids, k=1)
            internal_rels = [
                r for r in sub_rels if r.source_id in node_ids and r.target_id in node_ids
            ]

            # Derive cluster title
            title = self._infer_cluster_title(ents)

            comm = CommunitySummary(
                community_id=comm_id,
                title=title,
                entities=ents,
                relations=internal_rels
            )
            comm.generate_summary()
            self.communities[comm_id] = comm

        self.is_built = True
        return self.communities

    def _infer_cluster_title(self, entities: List[KGEntity]) -> str:
        """Dynamically infer thematic cluster title from constituent entity names and ontology types."""
        if not entities:
            return "Unassigned Cluster"

        # Count entity types
        type_counts: Dict[str, int] = defaultdict(int)
        for e in entities:
            type_counts[e.entity_type] += 1
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
        type_label = dominant_type.replace('_', ' ').title()

        # Find anchor entity names
        anchor_name = entities[0].name
        clean_anchor = re.sub(r'^(NovaCloud|Apple Inc|Acme)\s*', '', anchor_name, flags=re.I).strip()
        if not clean_anchor:
            clean_anchor = anchor_name

        if len(entities) == 1:
            return f"{clean_anchor} ({type_label})"
        elif len(entities) <= 3:
            names = [re.sub(r'^(NovaCloud|Apple Inc|Acme)\s*', '', e.name, flags=re.I).strip() for e in entities[:2]]
            return f"{' & '.join(names)} ({type_label}s)"
        else:
            return f"{clean_anchor} & Related {type_label}s ({len(entities)} nodes)"

    def global_search(self, query: str) -> Dict[str, Any]:
        """
        Execute Microsoft GraphRAG-style Global Search:
        Aggregates and synthesizes across all community summaries to answer macro-level questions.
        Ranks community relevance via dense ONNX vector cosine similarity.
        """
        if not self.is_built:
            self.build_communities()

        comm_list = list(self.communities.values())
        if comm_list:
            from ..retrieval.vector_store import DenseVectorStore
            v_store = DenseVectorStore()
            comm_texts = [f"{c.title}. {c.summary_text}" for c in comm_list]
            all_embs = v_store.embed_texts([query] + comm_texts)
            q_emb = all_embs[0]
            c_embs = all_embs[1:]
            sims = c_embs @ q_emb
            matched_summaries = [(float(sims[i]), comm_list[i]) for i in range(len(comm_list))]
            matched_summaries.sort(key=lambda x: x[0], reverse=True)
        else:
            matched_summaries = []

        lines = [
            f"### Global GraphRAG Synthesis for: \"{query}\"\n",
            "**Search Mode**: `GLOBAL_COMMUNITY_SEARCH` (Hierarchical Louvain Modularity Partitioning)\n",
            f"**Community Clusters Analyzed**: `{len(self.communities)}` total clusters\n"
        ]

        lines.append("#### Macro Community Landscape:")
        for score, comm in matched_summaries:
            lines.append(f"##### {comm.title}")
            lines.append(comm.summary_text)
            lines.append("")

        # If live LLM is configured, request live synthesis over community summaries
        if self.llm_client and self.llm_client.is_live_llm:
            comm_context = "\n\n".join(f"Community {c.community_id} ({c.title}):\n{c.summary_text}" for _, c in matched_summaries)
            live_res = self.llm_client.generate_synthesis(
                query=query,
                intent="GLOBAL_COMMUNITY_SEARCH",
                entities=[],
                relations=[],
                paths=[],
                evidence_text=f"Hierarchical Louvain Communities:\n{comm_context}"
            )
            return {
                "query": query,
                "mode": "GLOBAL_SEARCH",
                "communities_count": len(self.communities),
                "final_answer": live_res["answer"],
                "citations": [f"Community {c.community_id}: {c.title}" for _, c in matched_summaries]
            }

        lines.append("#### Cross-Cutting Synthesis & Strategic Assessment:")
        
        # Dynamically evaluate multi-community interactions
        fin_clusters = [c for _, c in matched_summaries if any(e.entity_type == "FINANCIAL_METRIC" for e in c.entities)]
        sys_clusters = [c for _, c in matched_summaries if any(e.entity_type == "SYSTEM_COMPONENT" for e in c.entities)]

        if fin_clusters and sys_clusters:
            fin_title = fin_clusters[0].title
            sys_title = sys_clusters[0].title
            lines.append(
                f"- **Cross-Domain Coupling ({fin_title} ↔ {sys_title})**: "
                f"Financial obligations in {fin_title} directly govern infrastructure investments and capacity bounds in {sys_title}."
            )
        
        for _, comm in matched_summaries[:3]:
            if comm.relations:
                sample_rel = comm.relations[0]
                lines.append(f"- **{comm.title} Intracluster Dynamic**: Internal linkage `{sample_rel.source_id}` ──[{sample_rel.relation_type}]──▶ `{sample_rel.target_id}`.")

        return {
            "query": query,
            "mode": "GLOBAL_SEARCH",
            "communities_count": len(self.communities),
            "final_answer": "\n".join(lines),
            "citations": [f"Community {c.community_id}: {c.title}" for _, c in matched_summaries]
        }
