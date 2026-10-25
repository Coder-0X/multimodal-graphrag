"""
Data schemas and ontology definitions for Multimodal GraphRAG.
Handles structured representations of financial statements, architectural diagrams,
entities, relations, and Knowledge Graph constructs.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Financial & Tabular Schemas
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    name: str = Field(..., description="Canonical metric name, e.g., 'Cash and cash equivalents'")
    raw_name: Optional[str] = None
    value: float = Field(..., description="Numerical value")
    currency: str = Field(default="USD")
    scale: str = Field(default="thousands", description="thousands, millions, or units")
    category: str = Field(..., description="e.g. Current Assets, Non-Current Liabilities, Revenue")
    line_number: Optional[int] = None
    footnotes: List[str] = Field(default_factory=list)


class FinancialStatement(BaseModel):
    statement_type: Literal["balance_sheet", "income_statement", "cash_flow", "footnote"]
    company_name: str
    reporting_period: str = Field(..., description="e.g. FY2025, Q3 2025")
    currency: str = "USD"
    scale: str = "thousands"
    line_items: List[LineItem] = Field(default_factory=list)
    computed_metrics: Dict[str, float] = Field(default_factory=dict, description="e.g. Current Ratio, Quick Ratio, Debt-to-Equity")
    raw_markdown: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. Architectural Diagram & Topology Schemas
# ---------------------------------------------------------------------------

class DiagramNode(BaseModel):
    id: str = Field(..., description="Unique component identifier, e.g., 'auth_service'")
    name: str = Field(..., description="Display name, e.g., 'Auth & Token Service'")
    component_type: str = Field(..., description="gateway, microservice, database, message_queue, cache, cdn")
    zone: Optional[str] = Field(default="internal_vpc", description="dmz, public, internal_vpc, db_tier")
    properties: Dict[str, Any] = Field(default_factory=dict)


class DiagramEdge(BaseModel):
    source_id: str
    target_id: str
    protocol: str = Field(default="HTTPS", description="HTTP, gRPC, TCP, Kafka Event, AMQP, CDC")
    sync_mode: Literal["sync", "async", "event_driven"] = "sync"
    description: Optional[str] = None
    port: Optional[int] = None
    is_critical_path: bool = False


class ArchitectureTopology(BaseModel):
    system_name: str
    environment: str = "Production"
    nodes: List[DiagramNode] = Field(default_factory=list)
    edges: List[DiagramEdge] = Field(default_factory=list)
    boundary_zones: List[str] = Field(default_factory=list)
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# 3. Knowledge Graph Ontologies
# ---------------------------------------------------------------------------

class KGEntity(BaseModel):
    id: str
    name: str
    entity_type: Literal[
        "ORGANIZATION",
        "FINANCIAL_METRIC",
        "SYSTEM_COMPONENT",
        "INFRASTRUCTURE_ZONE",
        "PROTOCOL",
        "RISK_FACTOR",
        "DOCUMENT_SECTION",
        "FISCAL_PERIOD",
        "CONCEPT"
    ]
    description: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_chunks: List[str] = Field(default_factory=list)


class KGRelation(BaseModel):
    source_id: str
    target_id: str
    relation_type: str = Field(
        ...,
        description="e.g. PART_OF, COMMUNICATES_WITH, CALLS, READS_FROM, DEPENDS_ON, AFFECTS, MITIGATES, REPORTS"
    )
    weight: float = 1.0
    description: str = ""
    evidence_snippet: Optional[str] = None


class KnowledgeGraphExport(BaseModel):
    entities: List[KGEntity]
    relations: List[KGRelation]
    metadata: Dict[str, Any] = Field(default_factory=dict)
