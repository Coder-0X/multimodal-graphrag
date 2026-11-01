"""
Architecture diagram extractor & topology analyzer.
Parses system components, boundary zones, protocols, and data flow topologies
into Knowledge Graph entities and directed relationships.
"""
import re
from typing import List, Tuple, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from .schemas import ArchitectureTopology, DiagramNode, DiagramEdge, KGEntity, KGRelation


# ---------------------------------------------------------------------------
# Formal Abstract Syntax Tree (AST) Data Structures for Architecture Diagrams
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    id: str
    name: str
    shape: str = "rectangle"
    zone: str = "Default Zone"
    component_type: str = "microservice"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ASTEdge:
    source_id: str
    target_id: str
    protocol: str = "HTTPS"
    sync_mode: str = "sync"
    arrow_style: str = "solid"  # solid, dotted, thick, bidirectional
    is_critical_path: bool = False
    description: Optional[str] = None


@dataclass
class ASTSubgraph:
    zone_id: str
    title: str
    parent_zone: Optional[str] = None
    nodes: List[str] = field(default_factory=list)


class DiagramASTParser:
    """
    Compiler-grade Lexer and Abstract Syntax Tree (AST) Parser for Mermaid,
    Graphviz DOT, and architecture flowchart specifications.
    Supports nested subgraphs, node shape tokens, arrow syntax variations,
    and multi-node chaining (A --> B --> C).
    """

    SHAPE_PATTERNS = [
        (r'\[\(([^\)]+)\)\]', "cylinder"),       # [(Database)]
        (r'\[\[([^\]]+)\]\]', "subroutine"),     # [[Subroutine]]
        (r'\(\[([^\]]+)\]\)', "stadium"),        # ([Stadium])
        (r'\(\(([^\)]+)\)\)', "circle"),         # ((Circle))
        (r'\{\{([^\}]+)\}\}', "hexagon"),        # {{Hexagon}}
        (r'\{([^\}]+)\}', "rhombus"),            # {Decision}
        (r'\[\/([^\/]+)\/\]', "parallelogram"),  # [/Data/]
        (r'\(([^\)]+)\)', "rounded"),            # (Rounded)
        (r'\[([^\]]+)\]', "rectangle"),          # [Rectangle]
    ]

    ARROW_REGEX = re.compile(
        r'(\s*(?:<==>|<-->|==>|-.->|-->|->|---|--)\s*(?:\|([^|]+)\|)?\s*)'
    )

    @classmethod
    def clean_text(cls, text: str) -> str:
        s = text.strip()
        while len(s) >= 2 and (
            (s.startswith('"') and s.endswith('"')) or
            (s.startswith("'") and s.endswith("'")) or
            (s.startswith('`') and s.endswith('`'))
        ):
            s = s[1:-1].strip()
        return s

    @classmethod
    def parse_node_token(cls, token: str) -> Tuple[str, str, str]:
        """
        Parses a node token into (node_id, display_name, shape).
        e.g., 'db[(PostgreSQL Cluster)]' -> ('db', 'PostgreSQL Cluster', 'cylinder')
        """
        token = token.strip()
        for pattern, shape in cls.SHAPE_PATTERNS:
            m = re.search(r'^([a-zA-Z0-9_\-]+)' + pattern + r'$', token)
            if m:
                nid = m.group(1).strip().lower()
                nlabel = cls.clean_text(m.group(2).strip())
                return nid, nlabel, shape

        # Standalone identifier without delimiter
        clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '', token).strip().lower()
        return clean_id, clean_id.replace('_', ' ').title(), "rectangle"

    @classmethod
    def parse(cls, raw_text: str, system_name: str = "Parsed Architecture Topology") -> ArchitectureTopology:
        nodes_dict: Dict[str, ASTNode] = {}
        edges: List[ASTEdge] = []
        zones: Set[str] = set()

        # Subgraph lexical scope stack
        scope_stack: List[str] = ["Default Zone"]

        lines = raw_text.splitlines()
        for line in lines:
            line_str = line.strip()
            # Ignore comments and diagram declarations
            if not line_str or line_str.startswith("%%") or line_str.startswith("```") or line_str.lower().startswith("flowchart") or line_str.lower().startswith("graph"):
                continue

            # Subgraph opening
            sub_m = re.match(r'subgraph\s+([a-zA-Z0-9_\-]+)?\s*(?:\[([^\]]+)\]|["\']([^"\']+)["\'])?', line_str, re.IGNORECASE)
            if sub_m:
                zone_title = sub_m.group(2) or sub_m.group(3) or sub_m.group(1) or "Perimeter Zone"
                zone_title = cls.clean_text(zone_title)
                zones.add(zone_title)
                scope_stack.append(zone_title)
                continue

            # Subgraph closure
            if line_str.lower() == "end":
                if len(scope_stack) > 1:
                    scope_stack.pop()
                continue

            current_zone = scope_stack[-1]

            # Check for Arrow Edges (including multi-segment chains: A --> B --> C)
            arrow_matches = list(cls.ARROW_REGEX.finditer(line_str))
            if arrow_matches:
                # Split line by arrows to extract node expressions
                node_exprs: List[str] = []
                last_idx = 0
                arrow_specs: List[Tuple[str, str]] = []  # (arrow_type, protocol)

                for am in arrow_matches:
                    node_exprs.append(line_str[last_idx:am.start()].strip())
                    raw_arrow = am.group(1).strip()
                    proto = am.group(2) or ""

                    # Protocol embedded in arrow? e.g. -- HTTPS -->
                    if not proto:
                        inner_p = re.search(r'--\s*([A-Za-z0-9_\-\/\s]+)\s*-->', raw_arrow)
                        if inner_p:
                            proto = inner_p.group(1).strip()

                    arrow_specs.append((raw_arrow, proto))
                    last_idx = am.end()

                node_exprs.append(line_str[last_idx:].strip())

                # Resolve all nodes in chain
                parsed_nodes: List[ASTNode] = []
                for n_expr in node_exprs:
                    if not n_expr:
                        continue
                    nid, nlabel, shape = cls.parse_node_token(n_expr)
                    if not nid:
                        continue
                    if nid not in nodes_dict:
                        ast_node = ASTNode(
                            id=nid,
                            name=nlabel,
                            shape=shape,
                            zone=current_zone,
                            component_type=ArchitectureDiagramExtractor._infer_component_type(nlabel)
                        )
                        nodes_dict[nid] = ast_node
                    else:
                        # Update label if token had an explicit shape and name
                        if nlabel != nid.replace('_', ' ').title():
                            nodes_dict[nid].name = nlabel
                            nodes_dict[nid].component_type = ArchitectureDiagramExtractor._infer_component_type(nlabel)
                            nodes_dict[nid].shape = shape
                    parsed_nodes.append(nodes_dict[nid])

                # Link consecutive pairs along the arrow chain
                for i in range(len(arrow_specs)):
                    if i < len(parsed_nodes) - 1:
                        src_ast = parsed_nodes[i]
                        tgt_ast = parsed_nodes[i + 1]
                        arr_token, proto_label = arrow_specs[i]

                        protocol = proto_label or "HTTPS"
                        is_critical = "==>" in arr_token
                        is_async = "-.->" in arr_token or any(q in protocol.lower() for q in ["kafka", "queue", "event", "amqp", "pubsub", "async"])

                        edges.append(ASTEdge(
                            source_id=src_ast.id,
                            target_id=tgt_ast.id,
                            protocol=protocol,
                            sync_mode="async" if is_async else "sync",
                            arrow_style="dotted" if "-.->" in arr_token else ("thick" if "==>" in arr_token else "solid"),
                            is_critical_path=is_critical,
                            description=f"Connects {src_ast.name} to {tgt_ast.name} via {protocol}"
                        ))
                continue

            # Standalone Node definition inside subgraph (e.g. node_id[Name])
            if any(sym in line_str for sym in ["[", "(", "{"]):
                nid, nlabel, shape = cls.parse_node_token(line_str)
                if nid and nid not in nodes_dict:
                    nodes_dict[nid] = ASTNode(
                        id=nid,
                        name=nlabel,
                        shape=shape,
                        zone=current_zone,
                        component_type=ArchitectureDiagramExtractor._infer_component_type(nlabel)
                    )

        # Convert AST structures to ArchitectureTopology
        topo_nodes: List[DiagramNode] = []
        for n in nodes_dict.values():
            topo_nodes.append(DiagramNode(
                id=n.id,
                name=n.name,
                component_type=n.component_type,
                zone=n.zone,
                properties={"shape": n.shape, "zone": n.zone}
            ))

        topo_edges: List[DiagramEdge] = []
        for e in edges:
            topo_edges.append(DiagramEdge(
                source_id=e.source_id,
                target_id=e.target_id,
                protocol=e.protocol,
                sync_mode=e.sync_mode,
                is_critical_path=e.is_critical_path,
                description=e.description
            ))

        return ArchitectureTopology(
            system_name=system_name,
            environment="Production",
            nodes=topo_nodes,
            edges=topo_edges,
            boundary_zones=list(zones) if zones else ["Default Zone"],
            description=f"AST-compiled topology with {len(topo_nodes)} components and {len(topo_edges)} communication pathways."
        )


class ArchitectureDiagramExtractor:
    """Extracts nodes, edges, network zones, and protocols from architectural diagrams via AST compiler."""

    @staticmethod
    def _clean_label(label: str) -> str:
        return DiagramASTParser.clean_text(label)

    @classmethod
    def extract_from_text(
        cls,
        raw_text: str,
        system_name: str = "Parsed Architecture Topology"
    ) -> ArchitectureTopology:
        """
        Parses Mermaid diagrams, Graphviz DOT, or text topology flowcharts into ArchitectureTopology
        via formal Abstract Syntax Tree (AST) compilation.
        """
        return DiagramASTParser.parse(raw_text=raw_text, system_name=system_name)


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
