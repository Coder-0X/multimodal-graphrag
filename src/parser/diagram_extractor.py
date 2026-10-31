"""
Architecture diagram extractor & topology analyzer.
Parses system components, boundary zones, protocols, and data flow topologies
into Knowledge Graph entities and directed relationships.
"""
import re
from typing import List, Tuple, Dict, Any, Set
from .schemas import ArchitectureTopology, DiagramNode, DiagramEdge, KGEntity, KGRelation


class ArchitectureDiagramExtractor:
    """Extracts nodes, edges, network zones, and protocols from architectural diagrams."""

    @staticmethod
    def _clean_label(label: str) -> str:
        """Strips surrounding brackets, parentheses, and quotes from node labels."""
        lbl = label.strip()
        while len(lbl) >= 2 and (
            (lbl.startswith('(') and lbl.endswith(')')) or
            (lbl.startswith('[') and lbl.endswith(']')) or
            (lbl.startswith('{') and lbl.endswith('}')) or
            (lbl.startswith('"') and lbl.endswith('"')) or
            (lbl.startswith("'") and lbl.endswith("'"))
        ):
            lbl = lbl[1:-1].strip()
        return lbl

    @classmethod
    def extract_from_text(
        cls,
        raw_text: str,
        system_name: str = "Parsed Architecture Topology"
    ) -> ArchitectureTopology:
        """
        Parses Mermaid diagrams, Graphviz DOT, or text topology flowcharts into ArchitectureTopology.
        Extracts component nodes, network zones, directional connections, and protocols.
        """
        nodes_dict: Dict[str, DiagramNode] = {}
        edges: List[DiagramEdge] = []
        zones: Set[str] = set()

        current_zone = "Default Zone"

        # Regex patterns for Mermaid & text flowcharts:
        # e.g.: node_id[Display Name] -->|Protocol| target_id[Target Name]
        # or: NodeA -> NodeB (Protocol)
        mermaid_edge_regex = re.compile(
            r'([a-zA-Z0-9_\-]+)(?:\[([^\]]+)\]|\(([^\)]+)\)|\{([^\}]+)\})?\s*[-=]+>(?:\|([^|]+)\|)?\s*([a-zA-Z0-9_\-]+)(?:\[([^\]]+)\]|\(([^\)]+)\)|\{([^\}]+)\})?'
        )
        simple_arrow_regex = re.compile(
            r'([a-zA-Z0-9_\-\s]+?)\s*[-=]+>\s*([a-zA-Z0-9_\-\s]+?)(?:\s*[\(\[]([A-Za-z0-9_\-\/\s]+)[\)\]])?$'
        )
        subgraph_regex = re.compile(r'subgraph\s+["\']?([^"\'\]]+)["\']?', re.IGNORECASE)

        for line in raw_text.splitlines():
            line_str = line.strip()
            if not line_str or line_str.startswith("```") or line_str.lower().startswith("flowchart") or line_str.lower().startswith("graph"):
                continue

            # Detect subgraphs / zones
            sub_match = subgraph_regex.search(line_str)
            if sub_match:
                current_zone = sub_match.group(1).strip()
                zones.add(current_zone)
                continue
            if line_str == "end":
                current_zone = "Default Zone"
                continue

            # 1. Try Mermaid edge match
            m_match = mermaid_edge_regex.search(line_str)
            if m_match:
                src_id = m_match.group(1).strip().lower()
                src_raw = m_match.group(2) or m_match.group(3) or m_match.group(4) or src_id.replace('_', ' ').title()
                src_name = cls._clean_label(src_raw)

                proto = (m_match.group(5) or "HTTPS").strip()

                tgt_id = m_match.group(6).strip().lower()
                tgt_raw = m_match.group(7) or m_match.group(8) or m_match.group(9) or tgt_id.replace('_', ' ').title()
                tgt_name = cls._clean_label(tgt_raw)

                # Register nodes
                if src_id not in nodes_dict:
                    nodes_dict[src_id] = DiagramNode(
                        id=src_id,
                        name=src_name,
                        component_type=cls._infer_component_type(src_name),
                        zone=current_zone
                    )
                else:
                    if src_name != src_id.replace('_', ' ').title():
                        nodes_dict[src_id].name = src_name
                        nodes_dict[src_id].component_type = cls._infer_component_type(src_name)

                if tgt_id not in nodes_dict:
                    nodes_dict[tgt_id] = DiagramNode(
                        id=tgt_id,
                        name=tgt_name,
                        component_type=cls._infer_component_type(tgt_name),
                        zone=current_zone
                    )
                else:
                    if tgt_name != tgt_id.replace('_', ' ').title():
                        nodes_dict[tgt_id].name = tgt_name
                        nodes_dict[tgt_id].component_type = cls._infer_component_type(tgt_name)

                edges.append(DiagramEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    protocol=proto,
                    sync_mode="async" if any(q in proto.lower() for q in ["kafka", "queue", "event", "amqp", "pubsub"]) else "sync",
                    description=f"Connects {src_name} to {tgt_name} via {proto}"
                ))
                continue

            # 2. Try Simple Arrow Match
            arr_match = simple_arrow_regex.match(line_str)
            if arr_match:
                raw_src = cls._clean_label(arr_match.group(1).strip())
                raw_tgt = cls._clean_label(arr_match.group(2).strip())
                proto = (arr_match.group(3) or "HTTPS").strip()

                src_id = re.sub(r'[^a-zA-Z0-9_]', '_', raw_src.lower()).strip('_')
                tgt_id = re.sub(r'[^a-zA-Z0-9_]', '_', raw_tgt.lower()).strip('_')

                if src_id and tgt_id:
                    if src_id not in nodes_dict:
                        nodes_dict[src_id] = DiagramNode(
                            id=src_id,
                            name=raw_src,
                            component_type=cls._infer_component_type(raw_src),
                            zone=current_zone
                        )
                    if tgt_id not in nodes_dict:
                        nodes_dict[tgt_id] = DiagramNode(
                            id=tgt_id,
                            name=raw_tgt,
                            component_type=cls._infer_component_type(raw_tgt),
                            zone=current_zone
                        )
                    edges.append(DiagramEdge(
                        source_id=src_id,
                        target_id=tgt_id,
                        protocol=proto,
                        sync_mode="async" if "async" in proto.lower() or "event" in proto.lower() else "sync",
                        description=f"Traffic from {raw_src} to {raw_tgt} over {proto}"
                    ))

        return ArchitectureTopology(
            system_name=system_name,
            environment="Production",
            nodes=list(nodes_dict.values()),
            edges=edges,
            boundary_zones=list(zones) if zones else ["Default Zone"],
            description=f"Parsed topology with {len(nodes_dict)} components and {len(edges)} communication pathways."
        )

    @staticmethod
    def _infer_component_type(name: str) -> str:
        nl = name.lower()
        if any(w in nl for w in ["database", "db", "postgres", "mysql", "aurora", "datastore", "mongo", "sqlite"]):
            return "database"
        elif any(w in nl for w in ["gateway", "ingress", "kong", "envoy", "router"]):
            return "api_gateway"
        elif any(w in nl for w in ["cdn", "waf", "cloudflare", "edge", "front door"]):
            return "cdn_edge"
        elif any(w in nl for w in ["queue", "kafka", "rabbit", "pubsub", "sqs", "event"]):
            return "message_queue"
        elif any(w in nl for w in ["cache", "redis", "memcached"]):
            return "cache"
        elif any(w in nl for w in ["frontend", "ui", "client", "app", "browser", "mobile"]):
            return "frontend_client"
        else:
            return "microservice"

    @classmethod
    def to_knowledge_graph(cls, topology: ArchitectureTopology) -> Tuple[List[KGEntity], List[KGRelation]]:
        """Converts an ArchitectureTopology into KG entities and relationships."""
        entities: List[KGEntity] = []
        relations: List[KGRelation] = []

        # 1. System Entity
        system_id = f"sys_{topology.system_name.lower().replace(' ', '_')}"
        entities.append(KGEntity(
            id=system_id,
            name=topology.system_name,
            entity_type="SYSTEM_COMPONENT",
            description=f"Overall system architecture: {topology.system_name} ({topology.environment})",
            properties={
                "environment": topology.environment,
                "description": topology.description or ""
            }
        ))

        # 2. Boundary Zones
        zone_ids: Dict[str, str] = {}
        for zone in topology.boundary_zones:
            zone_id = f"zone_{zone.lower().replace(' ', '_')}"
            zone_ids[zone] = zone_id
            entities.append(KGEntity(
                id=zone_id,
                name=f"Zone: {zone}",
                entity_type="INFRASTRUCTURE_ZONE",
                description=f"Network or infrastructure security perimeter: {zone}",
                properties={"zone_name": zone}
            ))
            relations.append(KGRelation(
                source_id=system_id,
                target_id=zone_id,
                relation_type="HAS_ZONE",
                description=f"System encompasses {zone}"
            ))

        # 3. Component Nodes
        node_map = {}
        for node in topology.nodes:
            node_id = f"comp_{node.id.lower()}"
            node_map[node.id] = node_id
            
            entities.append(KGEntity(
                id=node_id,
                name=node.name,
                entity_type="SYSTEM_COMPONENT",
                description=f"{node.component_type.upper()} component in {node.zone or 'default'} zone. Properties: {node.properties}",
                properties={
                    "component_type": node.component_type,
                    "zone": node.zone,
                    **node.properties
                }
            ))

            # Relation: Component -> System (PART_OF_SYSTEM)
            relations.append(KGRelation(
                source_id=node_id,
                target_id=system_id,
                relation_type="PART_OF_SYSTEM",
                description=f"{node.name} is a component of {topology.system_name}"
            ))

            # Relation: Component -> Zone (LOCATED_IN)
            if node.zone and node.zone in zone_ids:
                relations.append(KGRelation(
                    source_id=node_id,
                    target_id=zone_ids[node.zone],
                    relation_type="LOCATED_IN",
                    description=f"{node.name} is deployed inside {node.zone}"
                ))

        # 4. Component Communication Edges & Protocols
        protocol_entities: Dict[str, str] = {}
        for edge in topology.edges:
            src_kg_id = node_map.get(edge.source_id, f"comp_{edge.source_id.lower()}")
            tgt_kg_id = node_map.get(edge.target_id, f"comp_{edge.target_id.lower()}")

            # Protocol Entity
            proto_name = edge.protocol.upper()
            if proto_name not in protocol_entities:
                proto_id = f"proto_{proto_name.lower().replace(' ', '_')}"
                protocol_entities[proto_name] = proto_id
                entities.append(KGEntity(
                    id=proto_id,
                    name=proto_name,
                    entity_type="PROTOCOL",
                    description=f"Communication transport protocol: {proto_name}",
                    properties={"protocol": proto_name}
                ))

            # Direct Relation between Components
            rel_type = "COMMUNICATES_WITH" if edge.sync_mode == "sync" else "EMITS_EVENTS_TO"
            desc = edge.description or f"Sends data over {edge.protocol} ({edge.sync_mode})"
            relations.append(KGRelation(
                source_id=src_kg_id,
                target_id=tgt_kg_id,
                relation_type=rel_type,
                weight=1.5 if edge.is_critical_path else 1.0,
                description=f"{desc} [Protocol: {edge.protocol}]",
                evidence_snippet=f"{edge.source_id} -> {edge.target_id} via {edge.protocol}"
            ))

            # Reverse dependency relation (TARGET depends on SOURCE for upstream input)
            relations.append(KGRelation(
                source_id=tgt_kg_id,
                target_id=src_kg_id,
                relation_type="DEPENDS_ON",
                weight=1.0,
                description=f"Receives upstream inputs from {src_kg_id}"
            ))

            # Link Source to Protocol
            relations.append(KGRelation(
                source_id=src_kg_id,
                target_id=protocol_entities[proto_name],
                relation_type="CONNECTS_VIA",
                description=f"Uses protocol {proto_name}"
            ))

        return entities, relations
