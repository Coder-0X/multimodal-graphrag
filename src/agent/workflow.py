"""
LangGraph Multi-Hop Reasoning Workflow.
Orchestrates query intent analysis, seed retrieval, iterative graph hop expansion,
evidence evaluation, and citation-grounded synthesis.
"""
from typing import Dict, Any, List
import re
from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .llm_client import LLMReasoningClient
from ..retrieval.hybrid_retriever import HybridRetriever
from ..graph.base import BaseGraphStore


class GraphRAGAgent:
    """LangGraph-driven Multi-Hop GraphRAG Agent with Grounded LLM Synthesis."""

    def __init__(self, retriever: HybridRetriever, graph_store: BaseGraphStore, api_key: str = None):
        self.retriever = retriever
        self.graph_store = graph_store
        self.llm_client = LLMReasoningClient(api_key=api_key)
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
            top_k_seeds=5,
            k_hops=2,
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
        # 2. For cross-modal queries, we have active bridging paths connecting both domains
        # 3. For path tracing, we have reached the terminal storage tier or max_hops
        is_sufficient = False
        if hop_count >= max_hops:
            is_sufficient = True
            reasoning.append(f"Maximum hop budget ({max_hops}) reached. Proceeding to synthesis.")
        elif intent == "CROSS_MODAL_FINANCIAL_INFRASTRUCTURE":
            if bridging_paths:
                is_sufficient = True
                reasoning.append(f"Resolved {len(bridging_paths)} cross-modal bridging paths connecting financial obligations to infrastructure.")
            else:
                reasoning.append(f"No bridging path found yet at hop {hop_count + 1}; expanding neighborhood...")
        elif intent == "ARCHITECTURE_DEPENDENCY_TRACING":
            # Check if any storage tier or terminal DB node is reached
            has_db = any(
                "postgres" in e.get("id", "").lower() or "storage" in e.get("properties", {}).get("zone", "").lower()
                for e in state["retrieved_subgraph_entities"]
            )
            if has_db and (bridging_paths or hop_count >= 1):
                is_sufficient = True
                reasoning.append("Multi-hop topology trace successfully reached the stateful storage tier.")
            else:
                reasoning.append(f"Tracing downstream path towards storage tier (currently at hop {hop_count + 1})...")
        elif len(relations) >= 8:
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
        """Synthesize final citation-grounded response via LLMReasoningClient."""
        query = state["query"]
        intent = state["intent"]
        entities = state["retrieved_subgraph_entities"]
        relations = state["retrieved_subgraph_relations"]
        paths = state["bridging_paths"]
        evidence_text = state.get("evidence_text", "")

        synthesis_result = self.llm_client.generate_synthesis(
            query=query,
            intent=intent,
            entities=entities,
            relations=relations,
            paths=paths,
            evidence_text=evidence_text
        )

        return {
            "final_answer": synthesis_result["answer"],
            "citations": synthesis_result["citations"],
            "engine": synthesis_result.get("engine", "Extractive Graph-Reasoner")
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
