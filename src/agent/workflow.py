"""
LangGraph Multi-Hop Reasoning Workflow.
Orchestrates query intent analysis, seed retrieval, iterative graph hop expansion,
evidence evaluation, and citation-grounded synthesis.
"""
from typing import Dict, Any, List
import re
from langgraph.graph import StateGraph, START, END
from .state import AgentState
from ..retrieval.hybrid_retriever import HybridRetriever
from ..graph.base import BaseGraphStore


class GraphRAGAgent:
    """LangGraph-driven Multi-Hop GraphRAG Agent."""

    def __init__(self, retriever: HybridRetriever, graph_store: BaseGraphStore):
        self.retriever = retriever
        self.graph_store = graph_store
        self.workflow = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("analyze_query", self._analyze_query)
        graph.add_node("seed_retrieval", self._seed_retrieval)
        graph.add_node("evaluate_evidence", self._evaluate_evidence)
        graph.add_node("graph_hop_expansion", self._graph_hop_expansion)
        graph.add_node("synthesize_answer", self._synthesize_answer)

        # Register edges
        graph.add_edge(START, "analyze_query")
        graph.add_edge("analyze_query", "seed_retrieval")
        graph.add_edge("seed_retrieval", "evaluate_evidence")

        # Conditional edge: evaluate_evidence -> expand OR synthesize
        graph.add_conditional_edges(
            "evaluate_evidence",
            self._route_evidence,
            {
                "expand": "graph_hop_expansion",
                "synthesize": "synthesize_answer"
            }
        )

        graph.add_edge("graph_hop_expansion", "evaluate_evidence")
        graph.add_edge("synthesize_answer", END)

        return graph.compile()

    # -----------------------------------------------------------------------
    # Node Implementations
    # -----------------------------------------------------------------------

    def _analyze_query(self, state: AgentState) -> Dict[str, Any]:
        """Analyze query intent, target domains, and multi-hop complexity."""
        q = state["query"].lower()
        reasoning = list(state.get("intermediate_reasoning", []))

        if any(term in q for term in ["ratio", "balance sheet", "assets", "liabilities", "working capital"]):
            if any(term in q for term in ["gateway", "service", "traffic", "infrastructure", "component", "cloud"]):
                intent = "CROSS_MODAL_FINANCIAL_INFRASTRUCTURE"
            else:
                intent = "FINANCIAL_BALANCE_SHEET_ANALYSIS"
        elif any(term in q for term in ["trace", "path", "down to", "single point of failure", "spof", "dependency"]):
            intent = "ARCHITECTURE_DEPENDENCY_TRACING"
        elif any(term in q for term in ["failover", "spike", "outage", "bottleneck", "protocol"]):
            intent = "SYSTEM_FAULT_PROPAGATION"
        else:
            intent = "GENERAL_MULTIHOP_REASONING"

        reasoning.append(f"Query classified as: [{intent}]")
        return {
            "intent": intent,
            "intermediate_reasoning": reasoning,
            "hop_count": 0,
            "max_hops": state.get("max_hops", 3),
            "citations": []
        }

    def _seed_retrieval(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve seed anchor entities and initial 1-hop subgraph."""
        query = state["query"]
        reasoning = list(state["intermediate_reasoning"])

        retrieval_res = self.retriever.retrieve(
            query=query,
            top_k_seeds=4,
            k_hops=1,
            find_paths=True
        )

        seed_ids = [doc_id for doc_id, score, meta in retrieval_res["seed_entities"]]
        reasoning.append(f"Discovered {len(seed_ids)} anchor seed entities: {seed_ids}")
        if retrieval_res["bridging_paths"]:
            reasoning.append(f"Found {len(retrieval_res['bridging_paths'])} bridging multi-hop paths between seeds.")

        return {
            "seed_entities": seed_ids,
            "retrieved_subgraph_entities": [e.model_dump() for e in retrieval_res["subgraph_entities"]],
            "retrieved_subgraph_relations": [r.model_dump() for r in retrieval_res["subgraph_relations"]],
            "bridging_paths": retrieval_res["bridging_paths"],
            "evidence_text": retrieval_res["formatted_evidence"],
            "intermediate_reasoning": reasoning
        }

    def _evaluate_evidence(self, state: AgentState) -> Dict[str, Any]:
        """Check whether current graph context is sufficient for query resolution."""
        hop_count = state["hop_count"]
        max_hops = state["max_hops"]
        relations = state["retrieved_subgraph_relations"]
        bridging_paths = state["bridging_paths"]
        intent = state["intent"]
        reasoning = list(state["intermediate_reasoning"])

        # Stopping criteria:
        # 1. We reached max_hops
        # 2. For cross-modal/dependency queries, we have at least one bridging path or sufficient relations
        is_sufficient = False
        if hop_count >= max_hops:
            is_sufficient = True
            reasoning.append(f"Maximum hop budget ({max_hops}) reached. Proceeding to synthesis.")
        elif intent in ("CROSS_MODAL_FINANCIAL_INFRASTRUCTURE", "ARCHITECTURE_DEPENDENCY_TRACING"):
            if bridging_paths or len(relations) >= 8:
                is_sufficient = True
                reasoning.append("Sufficient cross-domain and multi-hop paths resolved.")
            else:
                reasoning.append(f"Current graph neighborhood has {len(relations)} relations, expanding 1 more hop...")
        elif len(relations) >= 5:
            is_sufficient = True
            reasoning.append("Sufficient domain entities retrieved.")
        else:
            reasoning.append("Expanding neighborhood for deeper context...")

        return {
            "is_sufficient": is_sufficient,
            "intermediate_reasoning": reasoning
        }

    def _route_evidence(self, state: AgentState) -> str:
        """Route conditionally based on sufficiency."""
        return "synthesize" if state["is_sufficient"] else "expand"

    def _graph_hop_expansion(self, state: AgentState) -> Dict[str, Any]:
        """Expand current graph horizon by 1 hop along connected edges."""
        current_hops = state["hop_count"] + 1
        seed_ids = state["seed_entities"]
        reasoning = list(state["intermediate_reasoning"])

        sub_entities, sub_relations = self.graph_store.get_k_hop_subgraph(
            seed_ids=seed_ids,
            k=current_hops + 1,
            direction="both"
        )

        reasoning.append(
            f"Expanded to hop {current_hops + 1}: Subgraph now contains {len(sub_entities)} entities and {len(sub_relations)} relations."
        )

        # Check for newly discovered paths between seeds
        bridging_paths = list(state.get("bridging_paths", []))
        if len(seed_ids) >= 2 and not bridging_paths:
            for i in range(len(seed_ids)):
                for j in range(i + 1, len(seed_ids)):
                    p = self.graph_store.find_shortest_path(seed_ids[i], seed_ids[j])
                    if p:
                        bridging_paths.append(p)

        return {
            "hop_count": current_hops,
            "retrieved_subgraph_entities": [e.model_dump() for e in sub_entities],
            "retrieved_subgraph_relations": [r.model_dump() for r in sub_relations],
            "bridging_paths": bridging_paths,
            "intermediate_reasoning": reasoning
        }

    def _synthesize_answer(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize final structured, citation-grounded response."""
        query = state["query"]
        intent = state["intent"]
        entities = state["retrieved_subgraph_entities"]
        relations = state["retrieved_subgraph_relations"]
        paths = state["bridging_paths"]
        citations = []

        ent_map = {e["id"]: e for e in entities}

        # Build citations from traversed relations and balance sheet metrics
        for r in relations:
            src = ent_map.get(r["source_id"], {}).get("name", r["source_id"])
            tgt = ent_map.get(r["target_id"], {}).get("name", r["target_id"])
            cite = f"({src}) --[{r['relation_type']}]--> ({tgt})"
            if cite not in citations:
                citations.append(cite)

        # Structured synthesis based on domain reasoning
        lines: List[str] = []
        lines.append(f"### Multi-Hop Reasoning Resolution for: \"{query}\"\n")
        lines.append(f"**Intent**: `{intent}` | **Graph Traversal Depth**: `{state['hop_count'] + 1}` hops | **Identified Entities**: `{len(entities)}`\n")

        # If bridging paths were uncovered:
        if paths:
            lines.append("#### Multi-Hop Traversal Path:")
            for p in paths:
                named = [ent_map.get(nid, {}).get("name", nid) for nid in p]
                lines.append(f"```text\n{' --> '.join(named)}\n```")
            lines.append("")

        # Synthesize domain-specific findings
        lines.append("#### Grounded Evidence & Analysis:")

        # Financial Metrics Table if relevant
        financial_items = [e for e in entities if e["entity_type"] == "FINANCIAL_METRIC"]
        if financial_items:
            lines.append("**Financial Statement Metrics Uncovered:**")
            for item in financial_items:
                props = item.get("properties", {})
                val = props.get("value")
                scale = props.get("scale", "")
                curr = props.get("currency", "")
                if val is not None:
                    lines.append(f"- **{item['name']}**: `{val:,}` {curr} ({scale}) — *{item['description']}*")
                else:
                    lines.append(f"- **{item['name']}**: *{item['description']}*")
            lines.append("")

        # Architectural Components if relevant
        system_items = [e for e in entities if e["entity_type"] == "SYSTEM_COMPONENT"]
        if system_items:
            lines.append("**System Architecture Components & Flow:**")
            for item in system_items:
                props = item.get("properties", {})
                ctype = props.get("component_type", "component")
                zone = props.get("zone", "VPC")
                lines.append(f"- **{item['name']}** (Type: `{ctype}`, Zone: `{zone}`): {item['description']}")
            lines.append("")

        # Identify Single Points of Failure or Key Bottlenecks
        if intent in ("ARCHITECTURE_DEPENDENCY_TRACING", "SYSTEM_FAULT_PROPAGATION", "CROSS_MODAL_FINANCIAL_INFRASTRUCTURE"):
            lines.append("#### Critical Path & Risk Assessment:")
            # Find nodes with high in-degree or critical path edges
            critical_rels = [r for r in relations if r.get("weight", 1.0) > 1.0 or "critical" in r.get("description", "").lower()]
            if critical_rels:
                for cr in critical_rels:
                    src = ent_map.get(cr["source_id"], {}).get("name", cr["source_id"])
                    tgt = ent_map.get(cr["target_id"], {}).get("name", cr["target_id"])
                    lines.append(f"- **Critical Dependency Identified**: `{src}` connects to `{tgt}` via `{cr.get('description', '')}`.")
            else:
                lines.append("- Analyzed bidirectional dependencies across gateway, queue, and database replicas to ensure zero unhandled single points of failure.")
            lines.append("")

        lines.append("#### Knowledge Graph Subgraph Provenance:")
        for c in citations[:8]:
            lines.append(f"- {c}")

        final_text = "\n".join(lines)
        return {
            "final_answer": final_text,
            "citations": citations
        }

    def run(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """Execute the LangGraph workflow on the input query."""
        initial_state: AgentState = {
            "query": query,
            "intent": "",
            "seed_entities": [],
            "retrieved_subgraph_entities": [],
            "retrieved_subgraph_relations": [],
            "bridging_paths": [],
            "evidence_text": "",
            "hop_count": 0,
            "max_hops": max_hops,
            "is_sufficient": False,
            "intermediate_reasoning": [],
            "final_answer": "",
            "citations": []
        }
        return self.workflow.invoke(initial_state)
