"""
Knowledge Graph & Multimodal GraphRAG Pipeline.
Coordinates vision parsing of balance sheets & architecture diagrams,
builds the unified Knowledge Graph, links cross-modal dependencies,
and initializes the LangGraph multi-hop reasoning agent.
"""
from typing import Dict, Any, Tuple, Optional
import os
from .parser import (
    MultimodalVisionParser,
    FinancialTableExtractor,
    ArchitectureDiagramExtractor,
    FinancialStatement,
    ArchitectureTopology,
    KGEntity,
    KGRelation,
)
from .graph import (
    NetworkXGraphStore,
    Neo4jGraphStore,
    BaseGraphStore,
    HierarchicalCommunityEngine
)
from .retrieval import SimpleDenseVectorStore, HybridRetriever
from .agent import GraphRAGAgent


class MultimodalGraphRAGPipeline:
    """Unified pipeline for Multimodal Complex Document Ingestion & GraphRAG."""

    def __init__(self, use_neo4j: bool = False, neo4j_uri: str = "bolt://localhost:7687"):
        self.parser = MultimodalVisionParser()
        self.nx_store = NetworkXGraphStore()
        self.vector_store = SimpleDenseVectorStore()
        self.community_engine = HierarchicalCommunityEngine(graph_store=self.nx_store)
        
        # Dual-engine support
        self.use_neo4j = use_neo4j
        self.neo4j_store: Optional[Neo4jGraphStore] = None
        if use_neo4j:
            self.neo4j_store = Neo4jGraphStore(uri=neo4j_uri)
            self.graph_store: BaseGraphStore = self.neo4j_store if self.neo4j_store.is_connected else self.nx_store
        else:
            self.graph_store = self.nx_store

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            graph_store=self.graph_store
        )
        self.agent = GraphRAGAgent(
            retriever=self.retriever,
            graph_store=self.graph_store
        )

        self.statement: Optional[FinancialStatement] = None
        self.topology: Optional[ArchitectureTopology] = None

    def ingest_documents(
        self,
        balance_sheet_path: Optional[str] = None,
        diagram_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest both financial statements and architecture topologies into the unified graph."""
        # 1. Parse Financial Balance Sheet
        self.statement = self.parser.parse_balance_sheet(balance_sheet_path or "")
        f_entities, f_relations = FinancialTableExtractor.to_knowledge_graph(self.statement)

        # 2. Parse Architecture Diagram
        self.topology = self.parser.parse_architecture_diagram(diagram_path or "")
        a_entities, a_relations = ArchitectureDiagramExtractor.to_knowledge_graph(self.topology)

        # 3. Add to Graph Store
        for ent in f_entities + a_entities:
            self.nx_store.add_entity(ent)
        for rel in f_relations + a_relations:
            self.nx_store.add_relation(rel)

        # 4. Synthesize Cross-Modal Bridges
        cross_relations = self._create_cross_modal_links()
        for rel in cross_relations:
            self.nx_store.add_relation(rel)

        # 5. Populate Vector Store
        for ent in f_entities + a_entities:
            self.vector_store.add_entity(ent)

        # 6. If Neo4j is enabled and connected, sync full graph
        if self.use_neo4j and self.neo4j_store and self.neo4j_store.is_connected:
            self.neo4j_store.sync_from_networkx(self.nx_store)

        # 7. Compute Graph Analytics & Build Hierarchical Communities
        communities = self.community_engine.build_communities()
        centrality = self.nx_store.calculate_centrality()

        # Sort top components by PageRank
        top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_entities": len(self.nx_store._entities),
            "total_relations": len(self.nx_store._relations),
            "cross_modal_relations": len(cross_relations),
            "financial_metrics_count": len([e for e in f_entities if e.entity_type == "FINANCIAL_METRIC"]),
            "architecture_components_count": len([e for e in a_entities if e.entity_type == "SYSTEM_COMPONENT"]),
            "communities_detected": len(communities),
            "community_summaries": {cid: comm.title for cid, comm in communities.items()},
            "top_central_entities": [(self.nx_store.get_entity(nid).name if self.nx_store.get_entity(nid) else nid, round(score, 4)) for nid, score in top_hubs],
            "computed_financial_ratios": self.statement.computed_metrics
        }

    def ingest_apple_showcase(self) -> Dict[str, Any]:
        """Ingest Apple Inc. FY2024 Form 10-K and Private Cloud Compute Architecture."""
        data = MultimodalVisionParser.load_apple_inc_showcase()
        return self.ingest_custom_data(statement=data["statement"], topology=data["topology"])

    def ingest_custom_data(
        self,
        statement: Optional[FinancialStatement] = None,
        topology: Optional[ArchitectureTopology] = None
    ) -> Dict[str, Any]:
        """Ingest custom or user-uploaded financial statements and architectural topologies."""
        if statement:
            self.statement = statement
            f_entities, f_relations = FinancialTableExtractor.to_knowledge_graph(self.statement)
            for ent in f_entities:
                self.nx_store.add_entity(ent)
                self.vector_store.add_entity(ent)
            for rel in f_relations:
                self.nx_store.add_relation(rel)

        if topology:
            self.topology = topology
            a_entities, a_relations = ArchitectureDiagramExtractor.to_knowledge_graph(self.topology)
            for ent in a_entities:
                self.nx_store.add_entity(ent)
                self.vector_store.add_entity(ent)
            for rel in a_relations:
                self.nx_store.add_relation(rel)

        # Re-link and rebuild communities
        cross_relations = self._create_cross_modal_links()
        for rel in cross_relations:
            self.nx_store.add_relation(rel)

        communities = self.community_engine.build_communities()
        centrality = self.nx_store.calculate_centrality()
        top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_entities": len(self.nx_store._entities),
            "total_relations": len(self.nx_store._relations),
            "cross_modal_relations": len(cross_relations),
            "communities_detected": len(communities),
            "top_central_entities": [(self.nx_store.get_entity(nid).name if self.nx_store.get_entity(nid) else nid, round(score, 4)) for nid, score in top_hubs],
            "computed_financial_ratios": self.statement.computed_metrics if self.statement else {}
        }

    def _create_cross_modal_links(self) -> list[KGRelation]:
        """
        Synthesize cross-modal relationships connecting financial balance sheet line items
        to the underlying cloud infrastructure components that incur or govern them.
        """
        cross_links = []
        period = self.statement.reporting_period.lower() if self.statement else "fy2025"

        # Link: Cloud Hosting Commitments (Financial Liability) -> Payment Orchestrator (Compute Consumer)
        cross_links.append(KGRelation(
            source_id=f"metric_cloud_hosting_commitments_and_short_term_obligations_{period}",
            target_id="comp_payment_orchestrator",
            relation_type="FUNDS_COMPUTE_INFRASTRUCTURE",
            weight=2.0,
            description="Balance sheet cloud hosting liabilities directly fund Kubernetes compute for payment orchestration",
            evidence_snippet="10-K Footnote: AWS compute agreements support core payment transaction processing."
        ))

        # Link: Cloud Hosting Commitments -> Cloudflare CDN & WAF (Edge Security Provider)
        cross_links.append(KGRelation(
            source_id=f"metric_cloud_hosting_commitments_and_short_term_obligations_{period}",
            target_id="comp_cloudflare_edge",
            relation_type="FUNDS_EDGE_SECURITY",
            weight=1.8,
            description="Short-term cloud obligations include multi-year Cloudflare enterprise edge agreement",
            evidence_snippet="10-K Footnote: Includes commitments for global DDoS and CDN edge capacity."
        ))

        # Link: Cloud Lease Assets (Non-Current Asset) -> Stateful Storage Tier (PostgreSQL Aurora)
        cross_links.append(KGRelation(
            source_id=f"metric_cloud_data_center_operating_lease_assets_{period}",
            target_id="comp_postgres_primary",
            relation_type="HOSTS_DATA_ASSET",
            weight=1.5,
            description="Operating lease assets represent dedicated hardware hosting primary database storage",
            evidence_snippet="Balance sheet non-current asset: Cloud data center operating lease assets."
        ))

        return cross_links

    def query(self, user_question: str, max_hops: int = 2, search_mode: str = "auto") -> Dict[str, Any]:
        """
        Execute a query using either:
        - 'local': Multi-hop LangGraph traversal starting from dense vector seed anchors.
        - 'global': Microsoft GraphRAG-style synthesis across hierarchical community summaries.
        - 'auto': Automatically routes to global search for macro/thematic questions, else local search.
        """
        q_lower = user_question.lower()
        is_global = search_mode == "global" or (
            search_mode == "auto" and any(
                w in q_lower for w in ["macro", "systemic", "holistic", "overview", "all communities", "thematic", "overall posture"]
            )
        )

        if is_global:
            return self.community_engine.global_search(user_question)
        
        return self.agent.run(query=user_question, max_hops=max_hops)

    def global_query(self, user_question: str) -> Dict[str, Any]:
        """Directly query the global community summarization engine."""
        return self.community_engine.global_search(user_question)
