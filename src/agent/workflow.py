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
        """Analyze query intent, target domains, and multi-hop complexity via LLMReasoningClient."""
        query = state["query"]
        reasoning = list(state.get("intermediate_reasoning", []))

        analysis = self.llm_client.classify_intent(query)
        intent = analysis.get("intent", "GENERAL_MULTIHOP_REASONING")
        target_concepts = analysis.get("target_concepts", [])

        reasoning.append(f"Query classified as: [{intent}] (Target concepts: {target_concepts[:4]})")
        return {
            "intent": intent,
            "intermediate_reasoning": reasoning,
            "hop_count": 0,
            "max_hops": state.get("max_hops", 3),
            "citations": []
        }

    def _seed_retrieval(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve anchor seed entities and initial 1-hop seed neighborhood."""
        query = state["query"]
        reasoning = list(state["intermediate_reasoning"])

        # Fetch only immediate 1-hop local neighborhood without pre-solving multi-hop paths
        retrieval_res = self.retriever.retrieve(
            query=query,
            top_k_seeds=5,
            k_hops=1,
            find_paths=False
        )

        seed_ids = [doc_id for doc_id, score, meta in retrieval_res["seed_entities"]]
        reasoning.append(f"Hop 1: Discovered {len(seed_ids)} anchor seed entities: {seed_ids}")

        return {
            "seed_entities": seed_ids,
            "retrieved_subgraph_entities": [e.model_dump() for e in retrieval_res["subgraph_entities"]],
            "retrieved_subgraph_relations": [r.model_dump() for r in retrieval_res["subgraph_relations"]],
            "bridging_paths": [],
            "evidence_text": retrieval_res["formatted_evidence"],
            "intermediate_reasoning": reasoning,
            "hop_count": 1,
            "unmatched_concepts": []
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
        # 1. We reached max_hops budget
        # 2. For cross-modal queries: we have established bridging paths connecting financial & infrastructure domains
        # 3. For path tracing: we have reached the terminal stateful persistence tier or connected endpoints
        is_sufficient = False
        if hop_count >= max_hops:
            is_sufficient = True
            reasoning.append(f"Maximum hop budget ({max_hops}) reached. Proceeding to synthesis.")
        elif intent == "CROSS_MODAL_FINANCIAL_INFRASTRUCTURE":
            if bridging_paths:
                is_sufficient = True
                reasoning.append(f"Resolved {len(bridging_paths)} cross-modal bridging paths connecting financial obligations to infrastructure.")
            else:
                reasoning.append(f"Cross-modal bridge not yet resolved at hop {hop_count}; expanding frontier to connect domains...")
        elif intent == "ARCHITECTURE_DEPENDENCY_TRACING":
            terminal_types = {"database", "storage", "datastore", "cache", "data_store", "sink"}
            has_persistence = any(
                e.get("properties", {}).get("component_type", "").lower() in terminal_types
                or any(term in e.get("name", "").lower() for term in ["database", "storage", "postgres", "mysql", "aurora", "dynamo", "redis", "mongo", "data store"])
                or any(term in e.get("properties", {}).get("zone", "").lower() for term in ["storage", "persistence", "stateful", "data"])
                for e in state["retrieved_subgraph_entities"]
            )
            if has_persistence and (bridging_paths or hop_count >= 2):
                is_sufficient = True
                reasoning.append("Multi-hop topology trace successfully reached the stateful persistence tier.")
            else:
                reasoning.append(f"Tracing downstream path towards persistence tier (currently at hop {hop_count})...")
        elif len(relations) >= 12 or hop_count >= 2:
            is_sufficient = True
            reasoning.append(f"Sufficient domain context retrieved ({len(relations)} relations at hop {hop_count}).")
        else:
            reasoning.append(f"Expanding neighborhood for deeper context at hop {hop_count}...")

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
            k=current_hops,
            direction="both"
        )

        reasoning.append(
            f"Hop {current_hops}: Expanded subgraph to {len(sub_entities)} entities and {len(sub_relations)} relations."
        )

        # Check for newly discovered operational paths between query-grounded seeds
        bridging_paths = []
        intent = state.get("intent", "")
        q_words = set(re.findall(r'[a-zA-Z0-9]+', state["query"].lower()))

        def is_query_grounded(ent) -> bool:
            if not ent:
                return False
            ent_words = set(re.findall(r'[a-zA-Z0-9]+', ent.name.lower())) | set(re.findall(r'[a-zA-Z0-9]+', ent.id.lower()))
            common = ent_words.intersection(q_words) - {
                "the", "and", "for", "with", "from", "down", "sys", "comp", "metric", "cat",
                "org", "zone", "flow", "tier", "route", "customer", "payments", "cluster"
            }
            return len(common) > 0

        grounded_seeds = [s for s in seed_ids if is_query_grounded(self.graph_store.get_entity(s))]

        if len(grounded_seeds) >= 2:
            for i in range(len(grounded_seeds)):
                e1 = self.graph_store.get_entity(grounded_seeds[i])
                t1 = e1.entity_type if e1 else ""
                for j in range(i + 1, len(grounded_seeds)):
                    e2 = self.graph_store.get_entity(grounded_seeds[j])
                    t2 = e2.entity_type if e2 else ""

                    # For cross-modal queries: only validate paths that actually span disparate domains
                    if intent == "CROSS_MODAL_FINANCIAL_INFRASTRUCTURE":
                        is_cross_domain = (
                            (t1 == "FINANCIAL_METRIC" and t2 in ("SYSTEM_COMPONENT", "INFRASTRUCTURE_ZONE")) or
                            (t2 == "FINANCIAL_METRIC" and t1 in ("SYSTEM_COMPONENT", "INFRASTRUCTURE_ZONE"))
                        )
                        if not is_cross_domain:
                            continue

                    p = self.graph_store.find_shortest_path(grounded_seeds[i], grounded_seeds[j], operational_only=True)
                    if p and len(p) > 1:
                        bridging_paths.append(p)

            if bridging_paths:
                reasoning.append(f"Hop {current_hops}: Resolved {len(bridging_paths)} operational bridging paths between seeds.")

        return {
            "hop_count": current_hops,
            "retrieved_subgraph_entities": [e.model_dump() for e in sub_entities],
            "retrieved_subgraph_relations": [r.model_dump() for r in sub_relations],
            "bridging_paths": bridging_paths,
            "intermediate_reasoning": reasoning
        }

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
        unmatched = state.get("unmatched_concepts", [])

        synthesis_result = self.llm_client.generate_synthesis(
            query=query,
            intent=intent,
            entities=entities,
            relations=relations,
            paths=paths,
            evidence_text=evidence_text,
            unmatched_concepts=unmatched if len(unmatched) >= 2 else None
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
            "citations": [],
            "unmatched_concepts": []
        }
        return self.workflow.invoke(initial_state)
