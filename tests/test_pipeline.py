"""
Comprehensive Test Suite for Multimodal GraphRAG.
Tests:
- Financial table parsing & ratio computation
- Architectural diagram topology extraction
- NetworkX Knowledge Graph traversal & shortest paths
- Vector store similarity search
- LangGraph multi-hop reasoning workflow
"""
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import (
    MultimodalVisionParser,
    FinancialTableExtractor,
    ArchitectureDiagramExtractor,
    FinancialStatement,
    ArchitectureTopology,
)
from src.graph import NetworkXGraphStore
from src.retrieval import DenseVectorStore, SimpleDenseVectorStore, HybridRetriever
from src.agent import GraphRAGAgent
from src.pipeline import MultimodalGraphRAGPipeline


def test_financial_table_extractor():
    parser = MultimodalVisionParser()
    statement = parser.parse_balance_sheet("")
    assert statement.company_name == "NovaCloud Global Inc"
    assert len(statement.line_items) > 0

    ratios = FinancialTableExtractor.compute_ratios(statement)
    assert "current_ratio" in ratios
    assert ratios["current_ratio"] > 1.5
    assert "quick_ratio" in ratios
    assert "debt_to_equity_ratio" in ratios
    assert ratios["debt_to_equity_ratio"] > 0

    entities, relations = FinancialTableExtractor.to_knowledge_graph(statement)
    assert len(entities) >= len(statement.line_items)
    assert len(relations) >= len(statement.line_items)
    
    # Verify entity types
    types = {e.entity_type for e in entities}
    assert "ORGANIZATION" in types
    assert "FINANCIAL_METRIC" in types
    assert "FISCAL_PERIOD" in types


def test_architecture_diagram_extractor():
    parser = MultimodalVisionParser()
    topology = parser.parse_architecture_diagram("")
    assert topology.system_name == "NovaCloud Global Payment Platform"
    assert len(topology.nodes) >= 10
    assert len(topology.edges) >= 8

    entities, relations = ArchitectureDiagramExtractor.to_knowledge_graph(topology)
    assert len(entities) >= len(topology.nodes)
    
    types = {e.entity_type for e in entities}
    assert "SYSTEM_COMPONENT" in types
    assert "INFRASTRUCTURE_ZONE" in types
    assert "PROTOCOL" in types

    rel_types = {r.relation_type for r in relations}
    assert "COMMUNICATES_WITH" in rel_types or "EMITS_EVENTS_TO" in rel_types
    assert "DEPENDS_ON" in rel_types


def test_networkx_graph_store():
    store = NetworkXGraphStore()
    pipeline = MultimodalGraphRAGPipeline(use_neo4j=False)
    pipeline.ingest_documents()

    # Verify node count
    export = pipeline.nx_store.export()
    assert export.metadata["num_nodes"] >= 40
    assert export.metadata["num_edges"] >= 80

    # Shortest path test
    path = pipeline.nx_store.find_shortest_path(
        "comp_checkout_ui",
        "comp_postgres_primary"
    )
    assert len(path) >= 3
    assert path[0] == "comp_checkout_ui"
    assert path[-1] == "comp_postgres_primary"

    # Community detection
    communities = pipeline.nx_store.detect_communities()
    assert len(communities) >= 2

    # Centrality
    centrality = pipeline.nx_store.calculate_centrality()
    assert len(centrality) > 0
    assert "comp_payment_orchestrator" in centrality


def test_hybrid_retriever():
    pipeline = MultimodalGraphRAGPipeline(use_neo4j=False)
    pipeline.ingest_documents()

    # Search for balance sheet liability
    result = pipeline.retriever.retrieve(
        query="cloud hosting commitments obligations",
        top_k_seeds=3,
        k_hops=1
    )
    assert len(result["seed_entities"]) > 0
    assert len(result["subgraph_entities"]) > 0
    assert len(result["subgraph_relations"]) > 0


def test_langgraph_multi_hop_agent():
    pipeline = MultimodalGraphRAGPipeline(use_neo4j=False)
    pipeline.ingest_documents()

    # Test Cross-Modal Financial -> Infrastructure query
    res1 = pipeline.query(
        "If our AWS cloud hosting commitment liabilities spike due to payment gateway traffic surge, which system components are directly involved?",
        max_hops=2
    )
    assert res1["intent"] == "CROSS_MODAL_FINANCIAL_INFRASTRUCTURE"
    assert len(res1["seed_entities"]) > 0
    assert "final_answer" in res1
    assert len(res1["citations"]) > 0

    # Test Architecture path tracing query
    res2 = pipeline.query(
        "Trace the multi-hop dependency path from the User Checkout flow down to the storage tier",
        max_hops=3
    )
    assert res2["intent"] == "ARCHITECTURE_DEPENDENCY_TRACING"
    assert "User Checkout Web Frontend" in res2["final_answer"]
    assert "PostgreSQL" in res2["final_answer"]


def test_hierarchical_community_engine():
    pipeline = MultimodalGraphRAGPipeline(use_neo4j=False)
    pipeline.ingest_documents()

    assert pipeline.community_engine.is_built
    assert len(pipeline.community_engine.communities) >= 2

    # Global search query
    res = pipeline.global_query("Summarize overall system risks across infrastructure and the balance sheet")
    assert res["mode"] == "GLOBAL_SEARCH"
    assert "Macro Community Landscape" in res["final_answer"]
    assert len(res["citations"]) > 0


def test_benchmark_runner():
    from src.evaluation.benchmark import BenchmarkRunner, BENCHMARK_CASES, compute_statistical_report
    assert len(BENCHMARK_CASES) >= 15
    runner = BenchmarkRunner()
    case_res = runner.evaluate_case(BENCHMARK_CASES[0])

    assert case_res["graph_rag"]["completeness"] >= case_res["standard_rag"]["completeness"]
    assert case_res["graph_rag"]["cross_modal_captured"] is True

    stats = compute_statistical_report([case_res, case_res])
    assert stats["n"] == 2
    assert "p_value" in stats


def test_apple_showcase_ingestion():
    pipeline = MultimodalGraphRAGPipeline(use_neo4j=False)
    pipeline.ingest_documents()
    initial_count = len(pipeline.nx_store._entities)

    stats = pipeline.ingest_apple_showcase()
    assert len(pipeline.nx_store._entities) > initial_count
    assert "Apple Inc" in [e.name for e in pipeline.nx_store._entities.values()]
    assert pipeline.statement.company_name == "Apple Inc"
    assert "current_ratio" in pipeline.statement.computed_metrics


def test_dynamic_table_text_parser():
    raw_table = """
    | Consolidated Balance Sheet | Dec 31, 2025 |
    | Cash and cash equivalents | $5,000,000 |
    | Accounts receivable | $2,500,000 |
    | Total current assets | $10,000,000 |
    | Accounts payable | $2,000,000 |
    | Total current liabilities | $4,000,000 |
    | Total liabilities | $6,000,000 |
    | Total equity | $8,000,000 |
    """
    stmt = FinancialTableExtractor.extract_from_text(raw_table, company_name="Acme Corp", period="Q4 2025")
    assert stmt.company_name == "Acme Corp"
    assert len(stmt.line_items) >= 5
    assert stmt.computed_metrics["current_ratio"] == 2.5
    assert stmt.computed_metrics["quick_ratio"] == 1.875


def test_dynamic_diagram_text_parser():
    mermaid_text = """
    flowchart TD
        subgraph "Ingress Perimeter"
            client[Web Browser Client] -->|HTTPS| gateway[Kong API Gateway]
        end
        subgraph "Core App"
            gateway -->|gRPC| auth_svc[Auth Token Service]
            gateway -->|gRPC| order_svc[Order Processing Service]
        end
        subgraph "Storage Tier"
            order_svc -->|TCP| db[(PostgreSQL Cluster)]
        end
    """
    topo = ArchitectureDiagramExtractor.extract_from_text(mermaid_text, system_name="E-Commerce Architecture")
    assert topo.system_name == "E-Commerce Architecture"
    assert len(topo.nodes) >= 5
    assert len(topo.edges) >= 4
    node_names = [n.name for n in topo.nodes]
    assert "Web Browser Client" in node_names
    assert "Kong API Gateway" in node_names
    assert "PostgreSQL Cluster" in node_names


def test_dense_vector_store():
    from src.retrieval.vector_store import DenseVectorStore
    store = DenseVectorStore()
    assert store.is_dense is True
    embs = store.embed_texts(["Cloud computing architecture", "Corporate finance liabilities"])
    assert embs.shape == (2, 384)
    sim = store.compute_similarity("Cloud payment processing", "AWS compute agreements")
    assert sim > 0.3


def test_semantic_cross_modal_linker():
    from src.graph.entity_resolution import SemanticCrossModalLinker
    from src.retrieval.vector_store import DenseVectorStore
    v_store = DenseVectorStore()
    linker = SemanticCrossModalLinker(vector_store=v_store, similarity_threshold=0.45)
    
    pipeline = MultimodalGraphRAGPipeline(use_neo4j=False)
    pipeline.ingest_documents()
    links = linker.discover_cross_modal_links(list(pipeline.nx_store._entities.values()))
    assert len(links) > 0
    rel_types = {l.relation_type for l in links}
    assert any("FUNDS" in t or "HOSTS" in t or "ASSOCIATED" in t for t in rel_types)


def test_hipporag_and_steiner_tree():
    """Validates HippoRAG Personalized PageRank and Steiner Minimal Tree subgraph extraction."""
    pipeline = MultimodalGraphRAGPipeline(use_neo4j=False)
    pipeline.ingest_documents()

    # 1. Test HippoRAG Personalized PageRank
    seed_weights = {
        "comp_kong_gateway": 0.9,
        "comp_payment_orchestrator": 0.7
    }
    ppr_scores = pipeline.nx_store.personalized_pagerank(seed_weights, alpha=0.85)
    assert len(ppr_scores) > 0
    assert "comp_kong_gateway" in ppr_scores
    assert ppr_scores["comp_kong_gateway"] > 0
    total_ppr = sum(ppr_scores.values())
    assert 0.95 <= total_ppr <= 1.05

    # 2. Test Steiner Minimal Tree Subgraph Extraction
    terminals = ["comp_kong_gateway", "comp_postgres_primary"]
    st_entities, st_relations = pipeline.nx_store.extract_steiner_subgraph(terminals, operational_only=True)
    assert len(st_entities) >= 2
    st_ids = {e.id for e in st_entities}
    assert "comp_kong_gateway" in st_ids
    assert "comp_postgres_primary" in st_ids
    assert len(st_relations) >= 1


    # 3. Test HybridRetriever with HippoRAG and Steiner Tree enabled
    retrieval_res = pipeline.retriever.retrieve(
        "Which microservices handle payment ingress and database persistence?",
        top_k_seeds=3,
        use_hipporag=True,
        use_steiner_tree=True
    )
    assert "hipporag_activation" in retrieval_res
    assert len(retrieval_res["hipporag_activation"]) > 0


def test_ast_diagram_parser():
    """Validates compiler-grade AST diagram parsing with nested subgraphs, node shapes, and arrow chaining."""
    from src.parser.diagram_extractor import DiagramASTParser

    mermaid_code = """
    subgraph DMZ_Perimeter ["DMZ Ingress Zone"]
        client[Web Client] -->|HTTPS| gw[(Kong Gateway)]
    end
    subgraph Core_VPC ["Microservices Core"]
        gw ==>|gRPC Sync| orch([Payment Orchestrator])
        orch -.->|Kafka Event| audit{{Audit Consumer}}
    end
    """
    topo = DiagramASTParser.parse(mermaid_code, system_name="AST Test System")
    assert len(topo.nodes) == 4
    node_by_id = {n.id: n for n in topo.nodes}

    assert "client" in node_by_id
    assert "gw" in node_by_id
    assert "orch" in node_by_id
    assert "audit" in node_by_id

    # Verify zone scopes
    assert node_by_id["client"].zone == "DMZ Ingress Zone"
    assert node_by_id["gw"].zone == "DMZ Ingress Zone"
    assert node_by_id["orch"].zone == "Microservices Core"
    assert node_by_id["audit"].zone == "Microservices Core"

    # Verify shapes
    assert node_by_id["gw"].properties.get("shape") == "cylinder"
    assert node_by_id["orch"].properties.get("shape") == "stadium"
    assert node_by_id["audit"].properties.get("shape") == "hexagon"

    # Verify edge metadata
    assert len(topo.edges) == 3
    edge_map = {(e.source_id, e.target_id): e for e in topo.edges}
    assert ("client", "gw") in edge_map
    assert ("gw", "orch") in edge_map
    assert ("orch", "audit") in edge_map

    assert edge_map[("gw", "orch")].is_critical_path is True
    assert edge_map[("orch", "audit")].sync_mode == "async"


def test_wilcoxon_and_bootstrap_stats():
    """Validates Wilcoxon Signed-Rank Test and Bootstrap Confidence Interval calculations."""
    from src.evaluation.benchmark import compute_wilcoxon_signed_rank, compute_bootstrap_confidence_interval

    std_recalls = [50.0, 60.0, 70.0, 55.0, 40.0, 65.0, 80.0, 50.0, 45.0, 70.0]
    graph_recalls = [90.0, 100.0, 95.0, 85.0, 80.0, 95.0, 100.0, 90.0, 85.0, 100.0]

    wilc = compute_wilcoxon_signed_rank(std_recalls, graph_recalls)
    assert wilc["p_value"] < 0.01
    assert wilc["W_plus"] > wilc["W_minus"]

    deltas = [g - s for g, s in zip(graph_recalls, std_recalls)]
    ci_low, ci_high = compute_bootstrap_confidence_interval(deltas, n_resamples=1000, ci=0.95)
    assert ci_low > 20.0
    assert ci_high > ci_low


if __name__ == "__main__":
    test_financial_table_extractor()
    print("✓ test_financial_table_extractor passed", flush=True)
    test_architecture_diagram_extractor()
    print("✓ test_architecture_diagram_extractor passed", flush=True)
    test_networkx_graph_store()
    print("✓ test_networkx_graph_store passed", flush=True)
    test_hybrid_retriever()
    print("✓ test_hybrid_retriever passed", flush=True)
    test_langgraph_multi_hop_agent()
    print("✓ test_langgraph_multi_hop_agent passed", flush=True)
    test_hierarchical_community_engine()
    print("✓ test_hierarchical_community_engine passed", flush=True)
    test_benchmark_runner()
    print("✓ test_benchmark_runner passed", flush=True)
    test_apple_showcase_ingestion()
    print("✓ test_apple_showcase_ingestion passed", flush=True)
    test_dynamic_table_text_parser()
    print("✓ test_dynamic_table_text_parser passed", flush=True)
    test_dynamic_diagram_text_parser()
    print("✓ test_dynamic_diagram_text_parser passed", flush=True)
    test_dense_vector_store()
    print("✓ test_dense_vector_store passed", flush=True)
    test_semantic_cross_modal_linker()
    print("✓ test_semantic_cross_modal_linker passed", flush=True)
    test_hipporag_and_steiner_tree()
    print("✓ test_hipporag_and_steiner_tree passed", flush=True)
    test_ast_diagram_parser()
    print("✓ test_ast_diagram_parser passed", flush=True)
    test_wilcoxon_and_bootstrap_stats()
    print("✓ test_wilcoxon_and_bootstrap_stats passed", flush=True)
    print("All 15 test suites passed successfully!", flush=True)


