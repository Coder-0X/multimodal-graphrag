"""
LangGraph Agent State Definitions.
Defines typed state passed across nodes in the multi-hop reasoning workflow.
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    query: str
    intent: str
    seed_entities: List[str]
    retrieved_subgraph_entities: List[Dict[str, Any]]
    retrieved_subgraph_relations: List[Dict[str, Any]]
    bridging_paths: List[List[str]]
    evidence_text: str
    hop_count: int
    max_hops: int
    is_sufficient: bool
    intermediate_reasoning: List[str]
    final_answer: str
    citations: List[str]
