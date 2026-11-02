"""
Streamlit & PyVis Interactive Visual Explorer for Multimodal GraphRAG.
Provides 2D physics-based network visualization, step-by-step LangGraph traversal playback,
live multimodal file ingestion (Gemini 2.0 Flash / PDF), and dual-mode search.
"""
import sys
import os
import json
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import MultimodalGraphRAGPipeline
from src.parser import (
    MultimodalVisionParser,
    FinancialStatement,
    ArchitectureTopology,
    LineItem,
    DiagramNode,
    DiagramEdge,
)


st.set_page_config(
    page_title="Omni-Modal GraphRAG Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #666; margin-bottom: 1.5rem; }
    .metric-badge { background-color: #f0f4f8; padding: 0.4rem 0.8rem; border-radius: 8px; font-weight: 600; }
    .hop-trace { background-color: #262730; color: #00E676; padding: 0.8rem; border-radius: 6px; font-family: monospace; }
</style>
""", unsafe_allow_html=True)


if "pipeline" not in st.session_state:
    pipeline = MultimodalGraphRAGPipeline(use_neo4j=False)
    pipeline.ingest_documents()
    st.session_state["pipeline"] = pipeline
    st.session_state["active_dataset"] = "NovaCloud Global Inc (FY2025)"

pipeline = st.session_state["pipeline"]

stats = {
    "total_entities": len(pipeline.nx_store._entities),
    "total_relations": len(pipeline.nx_store._relations),
    "communities": len(pipeline.community_engine.communities),
    "ratios": pipeline.statement.computed_metrics if pipeline.statement else {}
}

# ---------------------------------------------------------------------------
# Sidebar Controls & Analytics
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🕸️ Knowledge Graph Specs")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Entities", stats["total_entities"])
col_s2.metric("Relations", stats["total_relations"])

st.sidebar.markdown(f"**Active Dataset**: `{st.session_state['active_dataset']}`")
st.sidebar.markdown(f"**Hierarchical Communities**: `{stats['communities']}` detected")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Live AI Model Settings")
sidebar_api_key = st.sidebar.text_input(
    "Gemini / OpenAI API Key:",
    type="password",
    placeholder="AIzaSy... or sk-...",
    help="Optional. Enables live Gemini 2.0 Flash or GPT-4o-mini generation. Leave blank to run local ONNX FastEmbed + extractive reasoner."
)

if sidebar_api_key:
    if sidebar_api_key.startswith("AIza"):
        pipeline.agent.llm_client.gemini_key = sidebar_api_key
        st.sidebar.success("✅ Gemini 2.0 Flash Live Active")
    else:
        pipeline.agent.llm_client.openai_key = sidebar_api_key
        st.sidebar.success("✅ OpenAI Live Active")
else:
    st.sidebar.info("⚡ Local Engine: ONNX FastEmbed (bge-small-en-v1.5) + Extractive Reasoner")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Ingested Financial Ratios")
if stats["ratios"]:
    for r_name, r_val in stats["ratios"].items():
        st.sidebar.markdown(f"- **{r_name.replace('_', ' ').title()}**: `{r_val}`")
else:
    st.sidebar.caption("No financial statement loaded.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Graph Physics Controls")
physics_enabled = st.sidebar.checkbox("Enable Physics Dynamics", value=True)
node_distance = st.sidebar.slider("Node Separation Distance", min_value=80, max_value=300, value=140)


# ---------------------------------------------------------------------------
# Graph Visualization Generator (PyVis)
# ---------------------------------------------------------------------------

def generate_pyvis_html(pipeline: MultimodalGraphRAGPipeline, highlighted_nodes: list = None) -> str:
    highlighted_nodes = highlighted_nodes or []
    net = Network(height="520px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)

    # Color mapping for entity types
    color_map = {
        "ORGANIZATION": "#AB47BC",
        "FISCAL_PERIOD": "#BA68C8",
        "FINANCIAL_METRIC": "#4CAF50",
        "SYSTEM_COMPONENT": "#29B6F6",
        "INFRASTRUCTURE_ZONE": "#FFA726",
        "PROTOCOL": "#FF7043",
        "CONCEPT": "#78909C"
    }

    # Add Nodes
    for ent in pipeline.nx_store._entities.values():
        is_highlighted = ent.id in highlighted_nodes
        color = "#FFEB3B" if is_highlighted else color_map.get(ent.entity_type, "#90A4AE")
        size = 28 if is_highlighted else (20 if ent.entity_type in ("ORGANIZATION", "SYSTEM_COMPONENT") else 14)
        
        props_str = "<br>".join(f"{k}: {v}" for k, v in ent.properties.items() if k != "footnotes")
        title_hover = f"<b>{ent.name}</b><br>Type: {ent.entity_type}<br>{ent.description}<br><br>{props_str}"

        net.add_node(
            ent.id,
            label=ent.name[:22] + ("..." if len(ent.name) > 22 else ""),
            title=title_hover,
            color=color,
            size=size,
            borderWidth=3 if is_highlighted else 1
        )

    # Add Edges
    for (u, v, rtype), rel in pipeline.nx_store._relations.items():
        is_cross_modal = "FUNDS" in rtype or "HOSTS" in rtype
        edge_color = "#E91E63" if is_cross_modal else ("#FFEB3B" if u in highlighted_nodes and v in highlighted_nodes else "#546E7A")
        width = 2.5 if is_cross_modal else (2.0 if u in highlighted_nodes and v in highlighted_nodes else 1.0)
        
        net.add_edge(
            u,
            v,
            title=f"{rtype}: {rel.description}",
            label=rtype,
            color=edge_color,
            width=width,
            arrows="to"
        )

    if physics_enabled:
        net.barnes_hut(gravity=-2500, central_gravity=0.3, spring_length=node_distance, spring_strength=0.04)
    else:
        net.toggle_physics(False)

    return net.generate_html()


# ---------------------------------------------------------------------------
# Main View & Navigation Tabs
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">Omni-Modal GraphRAG Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimodal Complex Document Ingestion & Multi-Hop Reasoning over Knowledge Graphs</div>', unsafe_allow_html=True)

tab_explore, tab_ingest, tab_benchmark, tab_about = st.tabs([
    "🔍 Interactive Query & Graph",
    "📤 Ingest Real Document (Vision / PDF)",
    "📊 Empirical Benchmark",
    "📖 Architecture & Ontologies"
])

# ---------------------------------------------------------------------------
# Tab 1: Interactive Query & Traversal Playback
# ---------------------------------------------------------------------------

with tab_explore:
    sample_queries = [
        "If our AWS cloud hosting commitment liabilities spike due to payment gateway traffic surge, which system components are directly involved and what is the failover risk?",
        "What is the current ratio of the company, and which infrastructure cost centers contribute to the short-term operating obligations?",
        "Trace the multi-hop dependency path from the User Checkout flow down to the storage tier, identifying any single points of failure.",
        "Provide an overarching macro summary of all system risks across infrastructure and the balance sheet."
    ]

    col_q1, col_q2 = st.columns([3, 1])
    selected_sample = col_q1.selectbox("Select a benchmark multi-hop query or type below:", sample_queries)
    search_mode = col_q2.selectbox("Reasoning Strategy:", ["Auto-Route", "Local Multi-Hop (LangGraph)", "Global Search (Louvain Communities)"])

    user_query = st.text_input("Active Query:", value=selected_sample)

    highlighted_nids = []
    reasoning_result = None

    if st.button("🚀 Execute GraphRAG Reasoning", type="primary"):
        with st.spinner("Traversing knowledge graph & executing multi-hop synthesis..."):
            mode_param = "global" if "Global" in search_mode else ("local" if "Local" in search_mode else "auto")
            reasoning_result = pipeline.query(user_query, max_hops=3, search_mode=mode_param)

            if "retrieved_subgraph_entities" in reasoning_result:
                highlighted_nids = [e["id"] for e in reasoning_result["retrieved_subgraph_entities"]]
            elif "citations" in reasoning_result:
                highlighted_nids = []

    # Display Visual Graph Canvas
    st.markdown("### 🕸️ Knowledge Graph Topology Canvas")
    st.caption("Drag nodes, zoom, and inspect color-coded entities (Pink: Cross-Modal Links, Yellow: Traversed Entities).")
    
    html_graph = generate_pyvis_html(pipeline, highlighted_nodes=highlighted_nids)
    components.html(html_graph, height=540)

    # Display Agent Results
    if reasoning_result:
        st.markdown("---")
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            st.markdown("### 📝 Synthesized Multi-Hop Answer")
            st.markdown(reasoning_result.get("final_answer", ""))

        with res_col2:
            st.markdown("### 🧭 Traversal Provenance")
            if "intermediate_reasoning" in reasoning_result:
                st.markdown("**LangGraph State Machine Trace:**")
                for step in reasoning_result["intermediate_reasoning"]:
                    st.markdown(f"• `{step}`")

            if "citations" in reasoning_result:
                st.markdown("**Citations & Provenance Subgraphs:**")
                for c in reasoning_result["citations"][:6]:
                    st.markdown(f"- {c}")

# ---------------------------------------------------------------------------
# Tab 2: Live Multimodal Ingestion (Drag & Drop + Gemini Vision)
# ---------------------------------------------------------------------------

with tab_ingest:
    st.markdown("### 📤 Ingest Real Documents into the Knowledge Graph")
    st.markdown("Drop real financial PDFs (Form 10-K), balance sheet tables, or architectural diagram screenshots to dynamically expand the graph.")

    col_u1, col_u2 = st.columns([2, 1])

    with col_u1:
        uploaded_file = st.file_uploader(
            "Upload Document (PDF, Markdown 10-K, Mermaid Architecture, or Image):",
            type=["png", "jpg", "jpeg", "pdf", "md", "mmd", "txt"],
            help="Supports balance sheets (.md, .pdf), architecture blueprints (.mmd, .txt), or diagram images."
        )

        doc_type = st.radio(
            "Document Domain:",
            ["Financial Balance Sheet", "System Architecture Diagram", "Auto-Detect"],
            horizontal=True
        )

        api_key_input = st.text_input(
            "Gemini API Key (required only for image vision extraction):",
            type="password",
            placeholder="AIzaSy... (Markdown, Mermaid, and PDF parse 100% locally with zero keys)",
            help="Your key is used in-memory for this session only and is never stored."
        )

        if st.button("📥 Parse & Ingest Document", type="primary", disabled=uploaded_file is None):
            if uploaded_file:
                file_bytes = uploaded_file.read()
                file_name = uploaded_file.name.lower()

                with st.spinner("Extracting structured entities via Multimodal Parser..."):
                    try:
                        parser = MultimodalVisionParser(api_key=api_key_input or None)

                        if file_name.endswith(".pdf"):
                            statement = parser.extract_from_pdf(file_bytes)
                            ingest_stats = pipeline.ingest_custom_data(statement=statement)
                            st.session_state["active_dataset"] = f"Uploaded PDF: {uploaded_file.name}"
                            st.success(f"Successfully ingested PDF! Added {ingest_stats['total_entities']} entities.")
                            st.rerun()

                        elif file_name.endswith(".mmd") or (file_name.endswith(".txt") and doc_type == "System Architecture Diagram"):
                            text_content = file_bytes.decode("utf-8", errors="replace")
                            topology = ArchitectureDiagramExtractor.extract_from_text(text_content)
                            ingest_stats = pipeline.ingest_custom_data(topology=topology)
                            st.session_state["active_dataset"] = f"Architecture: {topology.system_name}"
                            st.success(f"Successfully ingested Architecture! Added {len(topology.nodes)} components.")
                            st.rerun()

                        elif file_name.endswith(".md") or (file_name.endswith(".txt") and doc_type == "Financial Balance Sheet"):
                            text_content = file_bytes.decode("utf-8", errors="replace")
                            statement = FinancialTableExtractor.extract_from_text(text_content)
                            ingest_stats = pipeline.ingest_custom_data(statement=statement)
                            st.session_state["active_dataset"] = f"10-K: {statement.company_name} ({statement.reporting_period})"
                            st.success(f"Successfully ingested Financial Table! Extracted {len(statement.line_items)} items.")
                            st.rerun()

                        elif file_name.endswith((".png", ".jpg", ".jpeg")):
                            key = api_key_input or sidebar_api_key
                            if not key:
                                st.warning(
                                    "Live visual extraction from raw images requires a Gemini API Key. "
                                    "Please enter your key above, or upload Markdown (.md) / Mermaid (.mmd) files for instant zero-key local parsing."
                                )
                            else:
                                mime = "image/png" if file_name.endswith(".png") else "image/jpeg"
                                extracted = parser.extract_from_image_gemini(file_bytes, mime_type=mime, api_key=key)
                                if extracted.get("type") == "balance_sheet" or "line_items" in extracted:
                                    statement = parser.dict_to_financial_statement(extracted)
                                    ingest_stats = pipeline.ingest_custom_data(statement=statement)
                                    st.session_state["active_dataset"] = f"Gemini Vision: {statement.company_name}"
                                    st.success(f"Ingested Balance Sheet via Gemini Vision! Extracted {len(statement.line_items)} line items.")
                                else:
                                    topology = parser.dict_to_architecture_topology(extracted)
                                    ingest_stats = pipeline.ingest_custom_data(topology=topology)
                                    st.session_state["active_dataset"] = f"Gemini Vision: {topology.system_name}"
                                    st.success(f"Ingested Architecture via Gemini Vision! Extracted {len(topology.nodes)} components.")
                                st.rerun()
                        else:
                            st.error(f"Unsupported file format: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")

    with col_u2:
        st.markdown("#### 🚀 Zero-Key Showcase Datasets")
        st.caption("Switch between pre-extracted real-world enterprise datasets:")

        if st.button("🏢 Load Apple Inc. (FY2024 Form 10-K & PCC Architecture)"):
            with st.spinner("Loading Apple Inc. 10-K & Private Cloud Compute graph..."):
                pipeline.ingest_apple_showcase()
                st.session_state["active_dataset"] = "Apple Inc. (FY2024 10-K + PCC)"
                st.success("Loaded Apple Inc. dataset! Balance sheet ratios and PCC secure enclave nodes added.")
                st.rerun()

        if st.button("☁️ Reset to NovaCloud Global (Payment Platform)"):
            with st.spinner("Resetting to NovaCloud Global platform..."):
                p_new = MultimodalGraphRAGPipeline(use_neo4j=False)
                p_new.ingest_documents()
                st.session_state["pipeline"] = p_new
                st.session_state["active_dataset"] = "NovaCloud Global Inc (FY2025)"
                st.success("Reset to default NovaCloud Global Platform!")
                st.rerun()

# ---------------------------------------------------------------------------
# Tab 3: Empirical Benchmark
# ---------------------------------------------------------------------------

with tab_benchmark:
    st.markdown("### 🏆 Empirical Research Benchmark: Traditional Vector RAG vs. Multimodal GraphRAG")
    st.markdown(
        "Scientifically rigorous evaluation measuring multi-hop recall, cross-modal edge discovery, "
        "and topological hallucination resistance with inferential hypothesis testing."
    )

    from src.evaluation.benchmark import BenchmarkRunner, compute_statistical_report, BENCHMARK_CASES

    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        eval_mode = st.radio(
            "Evaluation Corpus Scope:",
            ["Full Research Suite (N=20 Queries across 5 Categories)", "Fast Interactive Smoke Test (N=5 Queries)"],
            horizontal=True
        )
    with b_col2:
        st.caption("Includes: Cross-Modal Reasoning, Architecture Tracing, Financial Ratios, Global Search, and Negative Controls.")

    max_c = 20 if "Full" in eval_mode else 5

    if st.button(f"▶ Run Live Benchmark Suite ({max_c} Cases)"):
        with st.spinner(f"Evaluating Standard Vector RAG vs. Multimodal GraphRAG across {max_c} test cases..."):
            runner = BenchmarkRunner()
            b_results = runner.run_all(max_cases=max_c)
            stats = compute_statistical_report(b_results)

            # Metric cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("GraphRAG Recall", f"{stats['mean_graph']}%")
            m2.metric("Standard Vector RAG", f"{stats['mean_std']}%")
            m3.metric("Recall Gain (Δ)", f"+{stats['mean_delta']}%")
            p_disp = "< 0.001" if stats['p_value'] < 0.001 else f"{stats['p_value']:.4f}"
            m4.metric("t-test p-value", p_disp, delta="p < 0.05" if stats["is_significant"] else "p >= 0.05")

            # Significance banner
            if stats.get("is_significant"):
                st.success(
                    f"✅ **Statistically Significant Advantage (p = {p_disp}, t = {stats['t_stat']}, df = {stats['df']})**: "
                    f"Reject H₀ at α = 0.01 with 95% CI [{stats['ci_95'][0]}%, {stats['ci_95'][1]}%]. "
                    f"Cohen's d = {stats['cohen_d']} confirms a large effect size."
                )
            else:
                st.warning(f"p = {p_disp}: Insufficient evidence to reject H₀ at α = 0.05.")

            # Category Breakdown
            st.markdown("#### 📊 Category-Wise Aggregate Recall & Cross-Modal Recovery")
            cat_summary = []
            categories = sorted(list({r["category"] for r in b_results}))
            for cat in categories:
                c_res = [r for r in b_results if r["category"] == cat]
                c_n = len(c_res)
                c_std = sum(r["standard_rag"]["completeness"] for r in c_res) / c_n
                c_graph = sum(r["graph_rag"]["completeness"] for r in c_res) / c_n
                c_bridge = sum(1 for r in c_res if r["graph_rag"]["cross_modal_captured"]) / c_n * 100
                cat_summary.append({
                    "Research Category": cat,
                    "Sample Size (N)": c_n,
                    "Vector RAG Recall": f"{c_std:.1f}%",
                    "GraphRAG Recall": f"{c_graph:.1f}%",
                    "Absolute Gain (Δ)": f"+{c_graph - c_std:.1f}%",
                    "Cross-Modal Recovery": f"{c_bridge:.0f}%"
                })
            st.dataframe(cat_summary, use_container_width=True)

            # Detailed table
            st.markdown("#### 🔬 Detailed Case-by-Case Empirical Breakdown")
            table_data = []
            for r in b_results:
                table_data.append({
                    "Case ID": r["case_id"],
                    "Category": r["category"],
                    "Evaluation Query Title": r["title"],
                    "Vector RAG": f"{r['standard_rag']['completeness']}% ({r['standard_rag']['found_count']})",
                    "GraphRAG": f"{r['graph_rag']['completeness']}% ({r['graph_rag']['found_count']})",
                    "Cross-Modal": "✅ Supported" if r["graph_rag"]["cross_modal_captured"] else "❌ Failed",
                    "Hops": f"{r['graph_rag']['hops_traversed']} hops"
                })
            st.dataframe(table_data, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Architecture & Ontologies
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown("### 🏛️ Multimodal GraphRAG Architecture")
    st.markdown("""
    - **Multimodal Ingestion**: Converts complex Form 10-K balance sheets and microservice architecture diagrams into typed property graphs.
    - **Dual Storage**: Fast, zero-dependency in-memory **NetworkX** for local exploration, with enterprise **Neo4j** Cypher synchronization.
    - **Hybrid Retrieval**: Combines dense ONNX vector search (FastEmbed BAAI/bge-small-en-v1.5) with $k$-hop graph expansion and bridging path discovery.
    - **LangGraph Agent**: Dynamic state machine that assesses evidence completeness before synthesizing citation-backed answers.
    """)
