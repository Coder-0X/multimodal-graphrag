"""
Neo4j Knowledge Graph Store connector.
Provides Cypher query execution, labeled property graph mapping,
k-hop neighborhood expansion, and seamless migration from NetworkX.
"""
from typing import List, Dict, Optional, Tuple, Any
import logging
from .base import BaseGraphStore
from ..parser.schemas import KGEntity, KGRelation, KnowledgeGraphExport

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class Neo4jGraphStore(BaseGraphStore):
    """Production Neo4j Graph Store with Cypher traversal and property indexing."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None
        self._is_active = False

        if NEO4J_AVAILABLE:
            try:
                self._driver = GraphDatabase.driver(uri, auth=(user, password))
                # Verify connectivity
                with self._driver.session(database=self.database) as session:
                    session.run("RETURN 1")
                self._is_active = True
                self._create_constraints()
            except Exception as e:
                logging.warning(f"Neo4j instance at {uri} not reachable: {e}. Operating in standby mode.")
                self._is_active = False

    @property
    def is_connected(self) -> bool:
        return self._is_active

    def close(self):
        if self._driver:
            self._driver.close()

    def _create_constraints(self):
        """Create uniqueness constraint and indexes for fast entity lookup."""
        if not self._is_active:
            return
        with self._driver.session(database=self.database) as session:
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (e:KGEntity)
                REQUIRE e.id IS UNIQUE
            """)

    def add_entity(self, entity: KGEntity) -> None:
        if not self._is_active:
            return
        query = """
            MERGE (e:KGEntity {id: $id})
            SET e.name = $name,
                e.entity_type = $entity_type,
                e.description = $description,
                e.properties = $properties_json
        """
        import json
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                properties_json=json.dumps(entity.properties)
            )

    def add_relation(self, relation: KGRelation) -> None:
        if not self._is_active:
            return
        # Sanitize relation type for Cypher label
        clean_rel = relation.relation_type.replace(" ", "_").upper()
        query = f"""
            MERGE (a:KGEntity {{id: $source_id}})
            MERGE (b:KGEntity {{id: $target_id}})
            MERGE (a)-[r:{clean_rel}]->(b)
            SET r.weight = $weight,
                r.description = $description,
                r.evidence = $evidence
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                source_id=relation.source_id,
                target_id=relation.target_id,
                weight=relation.weight,
                description=relation.description,
                evidence=relation.evidence_snippet or ""
            )

    def get_entity(self, entity_id: str) -> Optional[KGEntity]:
        if not self._is_active:
            return None
        import json
        query = "MATCH (e:KGEntity {id: $id}) RETURN e"
        with self._driver.session(database=self.database) as session:
            res = session.run(query, id=entity_id).single()
            if not res:
                return None
            node = res["e"]
            props = json.loads(node.get("properties", "{}"))
            return KGEntity(
                id=node["id"],
                name=node["name"],
                entity_type=node["entity_type"],
                description=node["description"],
                properties=props
            )

    def get_k_hop_subgraph(
        self,
        seed_ids: List[str],
        k: int = 1,
        direction: str = "both"
    ) -> Tuple[List[KGEntity], List[KGRelation]]:
        if not self._is_active:
            return [], []
        
        arrow_in = "<-" if direction == "in" else "-"
        arrow_out = "->" if direction == "out" else "-"
        query = f"""
            MATCH (start:KGEntity)
            WHERE start.id IN $seeds
            MATCH path = (start){arrow_in}[r*1..{k}]{arrow_out}(neighbor:KGEntity)
            UNWIND nodes(path) AS n
            UNWIND relationships(path) AS rel
            RETURN DISTINCT n, rel
        """
        import json
        entities_dict = {}
        relations_list = []

        with self._driver.session(database=self.database) as session:
            result = session.run(query, seeds=seed_ids)
            for record in result:
                n = record["n"]
                if n["id"] not in entities_dict:
                    props = json.loads(n.get("properties", "{}"))
                    entities_dict[n["id"]] = KGEntity(
                        id=n["id"],
                        name=n["name"],
                        entity_type=n["entity_type"],
                        description=n["description"],
                        properties=props
                    )
                
                rel = record["rel"]
                relations_list.append(KGRelation(
                    source_id=rel.start_node["id"],
                    target_id=rel.end_node["id"],
                    relation_type=rel.type,
                    weight=rel.get("weight", 1.0),
                    description=rel.get("description", ""),
                    evidence_snippet=rel.get("evidence", "")
                ))

        return list(entities_dict.values()), relations_list

    def find_shortest_path(self, source_id: str, target_id: str) -> List[str]:
        if not self._is_active:
            return []
        query = """
            MATCH (a:KGEntity {id: $source_id}), (b:KGEntity {id: $target_id})
            MATCH p = shortestPath((a)-[*]-(b))
            RETURN [n in nodes(p) | n.id] AS path
        """
        with self._driver.session(database=self.database) as session:
            res = session.run(query, source_id=source_id, target_id=target_id).single()
            if res and res["path"]:
                return res["path"]
            return []

    def detect_communities(self) -> Dict[int, List[str]]:
        """Community detection placeholder using Louvain via APOC or fallback."""
        return {}

    def calculate_centrality(self) -> Dict[str, float]:
        """PageRank placeholder via Graph Data Science (GDS) or fallback."""
        return {}

    def export(self) -> KnowledgeGraphExport:
        return KnowledgeGraphExport(entities=[], relations=[])

    def sync_from_networkx(self, nx_store: BaseGraphStore) -> int:
        """Mirror an existing in-memory NetworkX store into Neo4j."""
        if not self._is_active:
            raise RuntimeError("Neo4j database is not connected.")
        export_data = nx_store.export()
        for ent in export_data.entities:
            self.add_entity(ent)
        for rel in export_data.relations:
            self.add_relation(rel)
        return len(export_data.entities)
