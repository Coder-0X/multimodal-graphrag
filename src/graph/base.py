"""
Abstract Base Interface for Knowledge Graph Stores.
Defines required methods for adding entities, querying k-hop subgraphs,
finding paths, and detecting communities.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from ..parser.schemas import KGEntity, KGRelation, KnowledgeGraphExport


class BaseGraphStore(ABC):
    """Abstract graph store interface for Multimodal GraphRAG."""

    @abstractmethod
    def add_entity(self, entity: KGEntity) -> None:
        """Add or update an entity node in the graph."""
        pass

    @abstractmethod
    def add_relation(self, relation: KGRelation) -> None:
        """Add or update a directed relationship edge between two entities."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[KGEntity]:
        """Retrieve an entity by its identifier."""
        pass

    @abstractmethod
    def get_k_hop_subgraph(
        self,
        seed_ids: List[str],
        k: int = 1,
        direction: str = "both"
    ) -> Tuple[List[KGEntity], List[KGRelation]]:
        """Retrieve all nodes and edges within k hops of the seed entity IDs."""
        pass

    @abstractmethod
    def find_shortest_path(self, source_id: str, target_id: str) -> List[str]:
        """Find the shortest sequence of entity IDs connecting source and target."""
        pass

    @abstractmethod
    def detect_communities(self) -> Dict[int, List[str]]:
        """Partition the graph into hierarchical or modular communities."""
        pass

    @abstractmethod
    def calculate_centrality(self) -> Dict[str, float]:
        """Calculate PageRank or degree centrality scores for nodes."""
        pass

    @abstractmethod
    def export(self) -> KnowledgeGraphExport:
        """Export the full graph into serialized entities and relations."""
        pass

    def personalized_pagerank(
        self,
        seed_weights: Dict[str, float],
        alpha: float = 0.85,
        max_iter: int = 100
    ) -> Dict[str, float]:
        """
        HippoRAG Associative Activation via Personalized PageRank (PPR).
        Diffuses activation probability across edge weights starting from seed entities.
        """
        return {}

    def extract_steiner_subgraph(
        self,
        terminal_ids: List[str]
    ) -> Tuple[List[KGEntity], List[KGRelation]]:
        """
        Extract the Steiner Minimal Tree (SMT) subgraph connecting all terminal entities
        with the minimal possible edge weight.
        """
        return [], []

