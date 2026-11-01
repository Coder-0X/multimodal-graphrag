"""
In-memory NetworkX Knowledge Graph Store.
Zero-dependency, high-performance graph algorithms including k-hop traversal,
shortest-path discovery, community detection, and PageRank.
"""
from typing import List, Dict, Optional, Tuple, Any, Set
import json
import networkx as nx
from .base import BaseGraphStore
from ..parser.schemas import KGEntity, KGRelation, KnowledgeGraphExport


class NetworkXGraphStore(BaseGraphStore):
    """In-memory Graph Store powered by NetworkX."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._entities: Dict[str, KGEntity] = {}
        self._relations: Dict[Tuple[str, str, str], KGRelation] = {}

    def add_entity(self, entity: KGEntity) -> None:
        """Add or update an entity node in the graph."""
        self._entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            properties=entity.properties,
            source_chunks=entity.source_chunks
        )

    def add_relation(self, relation: KGRelation) -> None:
        """Add or update a directed relationship edge between two entities."""
        # Ensure endpoints exist
        if relation.source_id not in self.graph:
            self.graph.add_node(relation.source_id, name=relation.source_id, entity_type="CONCEPT", description="", properties={})
        if relation.target_id not in self.graph:
            self.graph.add_node(relation.target_id, name=relation.target_id, entity_type="CONCEPT", description="", properties={})

        key = (relation.source_id, relation.target_id, relation.relation_type)
        self._relations[key] = relation
        
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            weight=relation.weight,
            description=relation.description,
            evidence_snippet=relation.evidence_snippet or ""
        )

    def get_entity(self, entity_id: str) -> Optional[KGEntity]:
        """Retrieve entity by ID."""
        return self._entities.get(entity_id)

    def get_k_hop_subgraph(
        self,
        seed_ids: List[str],
        k: int = 1,
        direction: str = "both"
    ) -> Tuple[List[KGEntity], List[KGRelation]]:
        """Retrieve all nodes and edges within k hops of the seed entity IDs."""
        visited_nodes: Set[str] = set()
        frontier: Set[str] = {s for s in seed_ids if s in self.graph}
        visited_nodes.update(frontier)

        for _ in range(k):
            next_frontier: Set[str] = set()
            for node in frontier:
                neighbors: Set[str] = set()
                if direction in ("out", "both"):
                    neighbors.update(self.graph.successors(node))
                if direction in ("in", "both"):
                    neighbors.update(self.graph.predecessors(node))
                
                for neighbor in neighbors:
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)
                        visited_nodes.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        # Collect entities
        subgraph_entities: List[KGEntity] = [
            self._entities[nid] for nid in visited_nodes if nid in self._entities
        ]

        # Collect relations spanning the visited nodes
        subgraph_relations: List[KGRelation] = []
        for (u, v, rtype), rel in self._relations.items():
            if u in visited_nodes and v in visited_nodes:
                subgraph_relations.append(rel)

        return subgraph_entities, subgraph_relations

    CONTAINER_EDGE_TYPES = {"PART_OF_SYSTEM", "RECORDED_IN", "REPORTS_FOR", "HAS_ACCOUNTING_SECTION", "HAS_ZONE"}

    def find_shortest_path(self, source_id: str, target_id: str, operational_only: bool = True) -> List[str]:
        """
        Find the shortest sequence of entity IDs connecting source and target.
        When operational_only is True, ignores structural/metadata container edges
        (e.g., PART_OF_SYSTEM, RECORDED_IN) to ensure paths reflect genuine
        operational dataflow, microservice dependencies, or cross-modal funding bridges.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return []

        # Build view with or without metadata container edges
        if operational_only:
            op_edges = [
                (u, v) for u, v, d in self.graph.edges(data=True)
                if d.get("relation_type") not in self.CONTAINER_EDGE_TYPES
            ]
            search_graph = self.graph.edge_subgraph(op_edges)
        else:
            search_graph = self.graph

        if source_id not in search_graph or target_id not in search_graph:
            return []

        try:
            # 1. Try directed path
            return nx.shortest_path(search_graph, source=source_id, target=target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # 2. Try undirected operational path (for bidirectional dependency resolution)
        try:
            undirected = search_graph.to_undirected()
            return nx.shortest_path(undirected, source=source_id, target=target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def detect_communities(self) -> Dict[int, List[str]]:
        """Partition the graph into modular communities using Louvain modularity optimization."""
        if len(self.graph) == 0:
            return {}
        undirected = self.graph.to_undirected()
        try:
            if hasattr(nx.community, "louvain_communities"):
                communities = list(nx.community.louvain_communities(undirected, seed=42))
            else:
                communities = list(nx.community.greedy_modularity_communities(undirected))
            return {i: list(comm) for i, comm in enumerate(communities)}
        except Exception:
            return {0: list(self.graph.nodes())}

    def calculate_centrality(self) -> Dict[str, float]:
        """Calculate PageRank scores for all nodes in the graph."""
        if len(self.graph) == 0:
            return {}
        try:
            return nx.pagerank(self.graph, weight="weight")
        except Exception:
            # Fallback to degree centrality
            return nx.degree_centrality(self.graph)

    def export(self) -> KnowledgeGraphExport:
        """Export the full graph into entities and relations."""
        return KnowledgeGraphExport(
            entities=list(self._entities.values()),
            relations=list(self._relations.values()),
            metadata={
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
            }
        )

    def save_json(self, filepath: str) -> None:
        """Serialize knowledge graph to a JSON file."""
        data = self.export().model_dump()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, filepath: str) -> None:
        """Load knowledge graph from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        export = KnowledgeGraphExport(**data)
        for entity in export.entities:
            self.add_entity(entity)
        for rel in export.relations:
            self.add_relation(rel)
