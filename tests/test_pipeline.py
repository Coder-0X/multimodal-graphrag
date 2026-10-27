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
from src.retrieval import SimpleDenseVectorStore, HybridRetriever
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
    from src.evaluation.benchmark import BenchmarkRunner, BENCHMARK_CASES
    runner = BenchmarkRunner()
    case_res = runner.evaluate_case(BENCHMARK_CASES[0])

    assert case_res["graph_rag"]["completeness"] >= case_res["standard_rag"]["completeness"]
    assert case_res["graph_rag"]["cross_modal_captured"] is True


if __name__ == "__main__":
    test_financial_table_extractor()
    test_architecture_diagram_extractor()
    test_networkx_graph_store()
    test_hybrid_retriever()
    test_langgraph_multi_hop_agent()
    test_hierarchical_community_engine()
    test_benchmark_runner()
    print("All 7 test suites passed successfully!")
