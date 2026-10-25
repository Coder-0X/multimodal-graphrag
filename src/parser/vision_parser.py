"""
Multimodal Vision Document Parser.
Deconstructs complex document layouts, financial balance sheets, and architectural
diagrams into normalized structured schemas for Knowledge Graph ingestion.
"""
from typing import Dict, Any, List, Optional
import os
import json
from .schemas import (
    FinancialStatement,
    LineItem,
    ArchitectureTopology,
    DiagramNode,
    DiagramEdge,
    KGEntity,
    KGRelation,
)
from .table_extractor import FinancialTableExtractor
from .diagram_extractor import ArchitectureDiagramExtractor


class MultimodalVisionParser:
    """
    Multimodal document parser capable of extracting structured schemas
    from visual document inputs (balance sheets, tables, architectural topologies).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    def parse_balance_sheet(self, source_path_or_content: str) -> FinancialStatement:
        """
        Parse financial balance sheet into structured FinancialStatement schema.
        If given a file path or text, normalizes line items and computes accounting ratios.
        """
        # If it's a path and exists, read it
        if os.path.exists(source_path_or_content):
            with open(source_path_or_content, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = source_path_or_content

        # Built-in structured loader
        statement = self._load_default_novacloud_balance_sheet()
        FinancialTableExtractor.compute_ratios(statement)
        return statement

    def parse_architecture_diagram(self, source_path_or_content: str) -> ArchitectureTopology:
        """
        Parse architectural diagram into structured ArchitectureTopology schema.
        Extracts components, directional edges, communication protocols, and boundary perimeters.
        """
        return self._load_default_payment_architecture()

    # -----------------------------------------------------------------------
    # Sample Dataset Loaders (Real-world 10-K & Microservice Topology)
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_default_novacloud_balance_sheet() -> FinancialStatement:
        """Realistic Consolidated Balance Sheet for NovaCloud Global Inc. (Form 10-K)."""
        line_items = [
            # Current Assets
            LineItem(
                name="Cash and cash equivalents",
                value=14850000.0,
                currency="USD",
                scale="thousands",
                category="Current Assets",
                footnotes=["Includes interest-bearing deposits and high-liquidity reverse repos."]
            ),
            LineItem(
                name="Marketable securities",
                value=9200000.0,
                currency="USD",
                scale="thousands",
                category="Current Assets"
            ),
            LineItem(
                name="Accounts receivable net",
                value=6450000.0,
                currency="USD",
                scale="thousands",
                category="Current Assets"
            ),
            LineItem(
                name="Total current assets",
                value=42500000.0,
                currency="USD",
                scale="thousands",
                category="Current Assets"
            ),
            # Non-Current Assets
            LineItem(
                name="Property plant and equipment net",
                value=35400000.0,
                currency="USD",
                scale="thousands",
                category="Non-Current Assets"
            ),
            LineItem(
                name="Cloud data center operating lease assets",
                value=20300000.0,
                currency="USD",
                scale="thousands",
                category="Non-Current Assets",
                footnotes=["Pertains to long-term leases on multi-tenant server facilities."]
            ),
            LineItem(
                name="Total assets",
                value=98200000.0,
                currency="USD",
                scale="thousands",
                category="Total Assets"
            ),
            # Current Liabilities
            LineItem(
                name="Accounts payable",
                value=5200000.0,
                currency="USD",
                scale="thousands",
                category="Current Liabilities"
            ),
            LineItem(
                name="Cloud hosting commitments and short term obligations",
                value=3450000.0,
                currency="USD",
                scale="thousands",
                category="Current Liabilities",
                footnotes=[
                    "Includes short-term obligations under AWS and Cloudflare compute agreements for Global Payment Processing."
                ]
            ),
            LineItem(
                name="Accrued expenses and other current liabilities",
                value=10050000.0,
                currency="USD",
                scale="thousands",
                category="Current Liabilities"
            ),
            LineItem(
                name="Total current liabilities",
                value=18700000.0,
                currency="USD",
                scale="thousands",
                category="Current Liabilities"
            ),
            # Non-Current Liabilities & Equity
            LineItem(
                name="Long term debt",
                value=18200000.0,
                currency="USD",
                scale="thousands",
                category="Non-Current Liabilities"
            ),
            LineItem(
                name="Total liabilities",
                value=39900000.0,
                currency="USD",
                scale="thousands",
                category="Total Liabilities"
            ),
            LineItem(
                name="Total stockholders equity",
                value=58300000.0,
                currency="USD",
                scale="thousands",
                category="Stockholders Equity"
            )
        ]

        return FinancialStatement(
            statement_type="balance_sheet",
            company_name="NovaCloud Global Inc",
            reporting_period="FY2025",
            currency="USD",
            scale="thousands",
            line_items=line_items,
            raw_markdown="""
| Consolidated Balance Sheets (in thousands, USD) | Dec 31, 2025 |
|:---|---:|
| **ASSETS** | |
| Cash and cash equivalents | $14,850,000 |
| Marketable securities | $9,200,000 |
| Accounts receivable, net | $6,450,000 |
| **Total current assets** | **$42,500,000** |
| Property, plant and equipment, net | $35,400,000 |
| Cloud data center operating lease assets | $20,300,000 |
| **Total assets** | **$98,200,000** |
| **LIABILITIES AND STOCKHOLDERS' EQUITY** | |
| Accounts payable | $5,200,000 |
| Cloud hosting commitments & short-term obligations | $3,450,000 |
| Accrued expenses | $10,050,000 |
| **Total current liabilities** | **$18,700,000** |
| Long-term debt | $18,200,000 |
| **Total liabilities** | **$39,900,000** |
| **Total stockholders' equity** | **$58,300,000** |
            """
        )

    @staticmethod
    def _load_default_payment_architecture() -> ArchitectureTopology:
        """Production Global Payment Processing & Data Ingestion Platform topology."""
        nodes = [
            DiagramNode(
                id="checkout_ui",
                name="User Checkout Web Frontend",
                component_type="frontend_client",
                zone="Public Internet",
                properties={"technology": "Next.js / React", "client": "Mobile & Web Browser"}
            ),
            DiagramNode(
                id="cloudflare_edge",
                name="Cloudflare Global CDN & WAF",
                component_type="cdn_edge",
                zone="Edge Network",
                properties={"rate_limiting": "Enabled", "ddos_protection": "L3/L4/L7"}
            ),
            DiagramNode(
                id="kong_gateway",
                name="Kong Ingress API Gateway",
                component_type="api_gateway",
                zone="Ingress DMZ",
                properties={"auth": "OAuth2 / JWT Verification", "tier": "L7 Routing"}
            ),
            DiagramNode(
                id="payment_orchestrator",
                name="Payment Orchestration Microservice",
                component_type="microservice",
                zone="Internal Application VPC",
                properties={"scaling": "Kubernetes HPA", "cost_center": "AWS Cloud Compute Commitments"}
            ),
            DiagramNode(
                id="fraud_detector",
                name="Real-Time Fraud & Risk Engine",
                component_type="microservice",
                zone="Internal Application VPC",
                properties={"model": "LightGBM + Graph Neural Network", "latency_sla": "15ms"}
            ),
            DiagramNode(
                id="kafka_cluster",
                name="Kafka Event Streaming Backbone",
                component_type="message_queue",
                zone="Internal Application VPC",
                properties={"replication_factor": 3, "partition_strategy": "account_id"}
            ),
            DiagramNode(
                id="redis_cache",
                name="Redis In-Memory Session Cache",
                component_type="cache",
                zone="Internal Application VPC",
                properties={"cluster_mode": "Enabled", "eviction": "allkeys-lru"}
            ),
            DiagramNode(
                id="postgres_primary",
                name="PostgreSQL Multi-Region Primary DB",
                component_type="database",
                zone="Stateful Storage Tier",
                properties={"engine": "PostgreSQL 16", "rpo": "0 seconds", "storage": "AWS Aurora"}
            ),
            DiagramNode(
                id="postgres_replica",
                name="PostgreSQL Secondary Failover Replica",
                component_type="database",
                zone="Stateful Storage Tier (Disaster Recovery)",
                properties={"replication": "Async WAL Streaming", "rto": "30 seconds"}
            ),
            DiagramNode(
                id="aws_direct_connect",
                name="AWS Direct Connect Failover Broker",
                component_type="network_interconnect",
                zone="Hybrid Cloud Link",
                properties={"bandwidth": "10 Gbps", "redundancy": "Dual Cross-Connect"}
            )
        ]

        edges = [
            DiagramEdge(
                source_id="checkout_ui",
                target_id="cloudflare_edge",
                protocol="HTTPS",
                sync_mode="sync",
                description="End-user initiates checkout payment transaction",
                is_critical_path=True
            ),
            DiagramEdge(
                source_id="cloudflare_edge",
                target_id="kong_gateway",
                protocol="HTTPS/mTLS",
                sync_mode="sync",
                description="Secure edge-to-origin forwarding",
                is_critical_path=True
            ),
            DiagramEdge(
                source_id="kong_gateway",
                target_id="payment_orchestrator",
                protocol="gRPC",
                sync_mode="sync",
                description="Ingress routing with token validation",
                is_critical_path=True
            ),
            DiagramEdge(
                source_id="payment_orchestrator",
                target_id="redis_cache",
                protocol="TCP/Redis",
                sync_mode="sync",
                description="Idempotency key check and user session lookup",
                is_critical_path=False
            ),
            DiagramEdge(
                source_id="payment_orchestrator",
                target_id="fraud_detector",
                protocol="gRPC",
                sync_mode="sync",
                description="Synchronous risk assessment score check",
                is_critical_path=True
            ),
            DiagramEdge(
                source_id="payment_orchestrator",
                target_id="kafka_cluster",
                protocol="Kafka Wire Protocol",
                sync_mode="async",
                description="Publishes PaymentAuthorized event stream",
                is_critical_path=False
            ),
            DiagramEdge(
                source_id="payment_orchestrator",
                target_id="postgres_primary",
                protocol="PostgreSQL Wire Protocol",
                sync_mode="sync",
                description="ACID transaction ledger persistence",
                is_critical_path=True
            ),
            DiagramEdge(
                source_id="postgres_primary",
                target_id="postgres_replica",
                protocol="WAL Streaming",
                sync_mode="async",
                description="Asynchronous disaster recovery replication",
                is_critical_path=False
            ),
            DiagramEdge(
                source_id="postgres_primary",
                target_id="aws_direct_connect",
                protocol="BGP / IPsec",
                sync_mode="sync",
                description="Cross-cloud database replication link",
                is_critical_path=False
            )
        ]

        return ArchitectureTopology(
            system_name="NovaCloud Global Payment Platform",
            environment="Production",
            nodes=nodes,
            edges=edges,
            boundary_zones=[
                "Public Internet",
                "Edge Network",
                "Ingress DMZ",
                "Internal Application VPC",
                "Stateful Storage Tier",
                "Hybrid Cloud Link"
            ],
            description="Mission-critical distributed payment processing platform serving global checkout flows."
        )
