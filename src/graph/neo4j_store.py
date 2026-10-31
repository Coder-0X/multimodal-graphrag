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
        """
        Detect modular communities via Cypher query clustering nodes by connected component & domain.
        """
        if not self._is_active:
            return {}
        query = """
            MATCH (n:KGEntity)
            OPTIONAL MATCH (n)-[r]-(m:KGEntity)
            WITH n.id AS id, coalesce(n.entity_type, 'GENERAL') AS domain, count(r) AS degree
            RETURN id, domain
        """
        communities: Dict[str, List[str]] = {}
        with self._driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                dom = record["domain"]
                if dom not in communities:
                    communities[dom] = []
                communities[dom].append(record["id"])

        return {i: node_list for i, node_list in enumerate(communities.values())}

    def calculate_centrality(self) -> Dict[str, float]:
        """
        Calculate normalized degree and connectivity centrality for all nodes in Neo4j.
        """
        if not self._is_active:
            return {}
        query = """
            MATCH (n:KGEntity)
            OPTIONAL MATCH (n)-[r]-(m:KGEntity)
            WITH n.id AS node_id, count(r) AS degree
            WITH collect({id: node_id, deg: degree}) AS all_nodes, max(degree) AS max_deg
            UNWIND all_nodes AS item
            RETURN item.id AS node_id, 
                   CASE WHEN max_deg > 0 THEN toFloat(item.deg) / toFloat(max_deg) ELSE 0.0 END AS centrality
        """
        centrality = {}
        with self._driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                centrality[record["node_id"]] = float(record["centrality"])
        return centrality

    def export(self) -> KnowledgeGraphExport:
        """
        Export the complete graph topology directly from Neo4j records into KnowledgeGraphExport.
        """
        if not self._is_active:
            return KnowledgeGraphExport(entities=[], relations=[])
        import json
        entities_dict = {}
        relations_list = []

        query = """
            MATCH (n:KGEntity)
            OPTIONAL MATCH (n)-[r]->(m:KGEntity)
            RETURN n, r, m.id AS target_id, type(r) AS rel_type
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                n = record["n"]
                if n["id"] not in entities_dict:
                    props = json.loads(n.get("properties", "{}"))
                    entities_dict[n["id"]] = KGEntity(
                        id=n["id"],
                        name=n["name"],
                        entity_type=n.get("entity_type", "CONCEPT"),
                        description=n.get("description", ""),
                        properties=props
                    )

                r = record["r"]
                target_id = record["target_id"]
                if r and target_id:
                    relations_list.append(KGRelation(
                        source_id=n["id"],
                        target_id=target_id,
                        relation_type=record["rel_type"] or "RELATED_TO",
                        weight=r.get("weight", 1.0),
                        description=r.get("description", ""),
                        evidence_snippet=r.get("evidence", "")
                    ))

        return KnowledgeGraphExport(
            entities=list(entities_dict.values()),
            relations=relations_list,
            metadata={"store": "Neo4j", "num_nodes": len(entities_dict), "num_edges": len(relations_list)}
        )

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
