# Multimodal GraphRAG: Complex Document Ingestion & Multi-Hop Reasoning Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Graph](https://img.shields.io/badge/KnowledgeGraph-NetworkX%20%7C%20Neo4j-green.svg)](https://networkx.org/)

An enterprise-grade, applied AI system that solves the fundamental limitations of traditional vector-only RAG when analyzing unstructured documents, multi-tier financial balance sheets, and complex architectural system diagrams.

---

## 🌟 What Makes It Stand Out

1. **Multimodal Complex Document Parsing**:
   - Parses structured tabular financial statements (Form 10-K balance sheets) into normalized entities and automatically computes derived accounting ratios (Current Ratio, Quick Ratio, Debt-to-Equity, Working Capital).
   - Deconstructs architectural system diagrams into typed components, boundary zones, directional network paths, and transport protocols.
2. **Dynamic Knowledge Graph Construction**:
   - Dual-engine architecture: In-memory **NetworkX** for local zero-dependency execution, alongside an enterprise **Neo4j** connector with Cypher traversal and property indexing.
   - Computes graph metrics: PageRank node centrality, betweenness, and modular community clustering.
   - **Cross-Modal Bridging**: Establishes explicit associative edges between financial liabilities (e.g. AWS hosting obligations) and physical infrastructure microservices.
3. **LangGraph Multi-Hop Reasoning**:
   - Replaces naive chunk retrieval with iterative graph traversal ($k$-hop neighborhood expansion + shortest-path discovery).
   - Resolves complex multi-hop queries that standard RAG fails on (e.g., propagating a financial cloud commitment surge to payment gateway failover risks).

---

## 🏗️ Architecture

```text
Complex Documents (10-K, Diagrams) 
                │
                ▼
  [Multimodal Vision Parser]
  ├── FinancialTableExtractor (Normalized Line Items + Computed Ratios)
  └── ArchitectureDiagramExtractor (Nodes, Directed Edges, Protocols, Zones)
                │
                ▼
     [Unified Knowledge Graph] ◄───► [Neo4j / NetworkX Store]
                │
        [Cross-Modal Linker]
                │
                ▼
    [Hybrid Retrieval Engine]
  ├── Dense Vector Store (Cosine / Subword Embeddings) -> Seed Anchors
  └── Graph Store Traversal (k-Hop Neighborhoods & Bridging Paths)
                │
                ▼
  [LangGraph Multi-Hop Agent]
  ├── Query Intent Classification
  ├── Iterative Graph Expansion Loop
  ├── Evidence Completeness Evaluator
  └── Citation-Grounded Synthesizer
```

---

## 📁 Project Directory Structure

```text
multimodal-graphrag/
├── requirements.txt            # Frozen dependency list
├── demo.py                     # Rich interactive CLI demo
├── README.md                   # System documentation
├── src/
│   ├── parser/
│   │   ├── schemas.py          # Pydantic schemas (Statements, Topologies, KG)
│   │   ├── table_extractor.py  # Financial table parser & ratio engine
│   │   ├── diagram_extractor.py# Architecture diagram topology parser
│   │   └── vision_parser.py    # Multimodal parser & sample datasets
│   ├── graph/
│   │   ├── base.py             # Abstract BaseGraphStore interface
│   │   ├── networkx_store.py   # In-memory graph engine (PageRank, Louvain)
│   │   └── neo4j_store.py      # Production Neo4j Cypher connector
│   ├── retrieval/
│   │   ├── vector_store.py     # Subword & TF-IDF dense vector store
│   │   └── hybrid_retriever.py # Hybrid Vector + Graph Traversal engine
│   ├── agent/
│   │   ├── state.py            # LangGraph agent state typing
│   │   └── workflow.py         # Stateful multi-hop reasoning state machine
│   └── pipeline.py             # Unified end-to-end orchestrator
└── tests/
    └── test_pipeline.py        # Automated test suite
```

---

## 🚀 Quick Start

### 1. Activate Environment

```bash
cd "C:\Users\NITRO V 15\.gemini\antigravity\scratch\multimodal-graphrag"
.\.venv\Scripts\activate
```

### 2. Run the Interactive Multi-Hop Demo

```bash
python demo.py
```

### 3. Run the Automated Test Suite

```bash
python tests/test_pipeline.py
```

---

## 💡 Benchmark Multi-Hop Queries Demonstrated

* **Query 1 (Cross-Modal Finance ↔ Infra)**:
  > *"If our AWS cloud hosting commitment liabilities spike due to payment gateway traffic surge, which system components are directly involved and what is the failover risk?"*
* **Query 2 (Balance Sheet Ratios & Obligations)**:
  > *"What is the current ratio of the company, and which infrastructure cost centers contribute to the short-term operating obligations?"*
* **Query 3 (End-to-End Architectural Path Tracing)**:
  > *"Trace the multi-hop dependency path from the User Checkout flow down to the storage tier, identifying any single points of failure."*

---

## 🔌 Connecting to Neo4j

To switch from the default in-memory NetworkX store to an active Neo4j database:

```python
from src.pipeline import MultimodalGraphRAGPipeline

pipeline = MultimodalGraphRAGPipeline(
    use_neo4j=True,
    neo4j_uri="bolt://localhost:7687"
)
pipeline.ingest_documents()
```
The pipeline automatically provisions constraints, labeled property graphs, and replicates NetworkX topologies directly into Neo4j Cypher records.
