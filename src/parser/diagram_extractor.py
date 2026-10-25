"""
Architecture diagram extractor & topology analyzer.
Parses system components, boundary zones, protocols, and data flow topologies
into Knowledge Graph entities and directed relationships.
"""
from typing import List, Tuple, Dict, Any
from .schemas import ArchitectureTopology, KGEntity, KGRelation


class ArchitectureDiagramExtractor:
    """Extracts nodes, edges, network zones, and protocols from architectural diagrams."""

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
