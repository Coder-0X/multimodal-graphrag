# Multimodal GraphRAG: Complex Document Ingestion & Multi-Hop Reasoning Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Graph](https://img.shields.io/badge/KnowledgeGraph-NetworkX%20%7C%20Neo4j-green.svg)](https://networkx.org/)
[![Streamlit UI](https://img.shields.io/badge/Explorer-Streamlit%20%2B%20PyVis-red.svg)](https://streamlit.io/)
[![CI](https://github.com/Coder-0X/multimodal-graphrag/actions/workflows/ci.yml/badge.svg)](https://github.com/Coder-0X/multimodal-graphrag/actions)

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
3. **Dual-Mode Retrieval (Local Multi-Hop + Global Community Search)**:
   - **Local Search**: Replaces naive chunk retrieval with iterative graph traversal ($k$-hop neighborhood expansion + shortest-path discovery).
   - **Global Search**: Microsoft GraphRAG-inspired hierarchical community detection (Leiden/Louvain clusters) to answer macro systemic risk queries.
4. **Interactive 2D Physics Visual Graph Explorer**:
   - Built with **Streamlit** and **PyVis** (WebGL) for real-time physics-based graph exploration and visual step-by-step LangGraph traversal playback.

---

## 📊 Empirical Benchmark: Traditional Vector RAG vs. Multimodal GraphRAG

In head-to-head empirical evaluations testing multi-hop entity recall, cross-modal linking, and numerical grounding on distributed enterprise documents:

| Query Evaluation Case | Standard Vector RAG | Multimodal GraphRAG | Cross-Modal Bridge Captured? | Graph Depth |
| :--- | :---: | :---: | :---: | :---: |
| **1. Cross-Modal Financial Liability $\rightarrow$ Infrastructure Propagation** | 66.7% (4/6) | **100.0% (6/6)** | **[YES] Supported** | 2 hops |
| **2. Balance Sheet Liquidity Ratios & Cost Allocation** | 40.0% (2/5) | **100.0% (5/5)** | **[YES] Supported** | 2 hops |
| **3. End-to-End Architectural Path Tracing (SPOF)** | 60.0% (3/5) | **100.0% (5/5)** | **[YES] Supported** *(Std: [NO])* | 2 hops |
| **Average Multi-Hop Score** | **55.6%** | **100.0%** | **100% Graph Capture** | **Adaptive** |

> **Why Traditional RAG Fails**: Standard vector search fragments documents into arbitrary windows. When a query requires linking a 10-K footnote (page 48) to an API Gateway (page 12) down to an Aurora failover replica (page 85), cosine similarity decays and chunk distance breaks the chain. **GraphRAG traverses topological edges regardless of document proximity.**

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
    [Cross-Modal Bridge Engine]
                │
        ┌───────┴────────────────────────┐
        ▼                                ▼
[Local Multi-Hop Search]        [Global Community Search]
(LangGraph + k-Hop Paths)       (Leiden Cluster Summaries)
        │                                │
        └───────┬────────────────────────┘
                ▼
      [Unified Synthesizer]
  ├── Interactive Visual UI (Streamlit + PyVis)
  └── Empirical Benchmark Suite
```

---

## 📁 Project Directory Structure

```text
multimodal-graphrag/
├── docker-compose.yml          # Production Neo4j + APOC stack
├── requirements.txt            # Frozen dependency list
├── app.py                      # Interactive Streamlit + PyVis Visual Explorer
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
│   │   ├── neo4j_store.py      # Production Neo4j Cypher connector
│   │   └── community.py        # Hierarchical Community Detection & Global Search
│   ├── retrieval/
│   │   ├── vector_store.py     # Subword & TF-IDF dense vector store
│   │   ├── standard_rag.py     # Traditional chunked vector baseline
│   │   └── hybrid_retriever.py # Hybrid Vector + Graph Traversal engine
│   ├── agent/
│   │   ├── state.py            # LangGraph agent state typing
│   │   └── workflow.py         # Stateful multi-hop reasoning state machine
│   ├── evaluation/
│   │   └── benchmark.py        # Empirical comparison suite (Vector vs GraphRAG)
│   └── pipeline.py             # Unified end-to-end orchestrator
└── tests/
    └── test_pipeline.py        # Automated 7-suite test pipeline
```

---

## 🚀 Quick Start

### 1. Activate Environment

```bash
cd "C:\Users\NITRO V 15\.gemini\antigravity\scratch\multimodal-graphrag"
.\.venv\Scripts\activate
```

### 2. Launch the Interactive Visual Graph Explorer

```bash
streamlit run app.py
```
Open `http://localhost:8501` to explore the 2D physics network, inspect highlighted traversal paths, and query local vs. global graph search.

### 3. Run the Head-to-Head Empirical Benchmark

```bash
python -m src.evaluation.benchmark
```

### 4. Run the Automated Test Suite

```bash
python tests/test_pipeline.py
```

---

## 🔌 Connecting to Enterprise Neo4j

### Spin up Neo4j via Docker Compose:
```bash
docker compose up -d
```
Access the Neo4j Browser at `http://localhost:7474` (Credentials: `neo4j` / `password`).

### Run the Pipeline with Neo4j Synchronization:
```python
from src.pipeline import MultimodalGraphRAGPipeline

pipeline = MultimodalGraphRAGPipeline(
    use_neo4j=True,
    neo4j_uri="bolt://localhost:7687"
)
pipeline.ingest_documents()
```
The pipeline automatically provisions constraints, labeled property graphs, and replicates NetworkX topologies directly into Neo4j Cypher records.
