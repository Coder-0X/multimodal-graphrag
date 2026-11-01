"""
Multimodal Vision Document Parser.
Deconstructs complex document layouts, financial balance sheets, and architectural
diagrams into normalized structured schemas for Knowledge Graph ingestion.
Supports reading authentic enterprise Markdown/Mermaid files from data/, live Gemini Vision API
extraction, and PDF extraction via pypdf.
"""
from typing import Dict, Any, List, Optional
import os
import json
import base64
import httpx
from io import BytesIO
import pypdf

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

# Path to the data directory containing real enterprise benchmark documents
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
)


class MultimodalVisionParser:
    """
    Multimodal document parser capable of extracting structured schemas
    from visual document inputs (balance sheets, tables, architectural topologies).
    Parses real documents from disk or calls Gemini 2.0 Flash Vision for live image extractions.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")

    def parse_balance_sheet(self, source_path_or_content: str = "") -> FinancialStatement:
        """
        Parse financial balance sheet into structured FinancialStatement schema.
        Reads authentic Markdown / PDF files from disk or parses provided text.
        """
        content = ""
        if source_path_or_content:
            if os.path.exists(source_path_or_content):
                if source_path_or_content.lower().endswith(".pdf"):
                    return self.extract_from_pdf(source_path_or_content, doc_type="balance_sheet")
                with open(source_path_or_content, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = source_path_or_content

        if not content:
            # Load and parse the authentic default NovaCloud 10-K document from disk
            default_file = os.path.join(DATA_DIR, "novacloud_fy2025_10k.md")
            if os.path.exists(default_file):
                with open(default_file, "r", encoding="utf-8") as f:
                    content = f.read()

        if content:
            parsed = FinancialTableExtractor.extract_from_text(
                content,
                company_name="NovaCloud Global Inc",
                period="FY2025"
            )
            if parsed.line_items:
                return parsed

        raise FileNotFoundError(f"Could not load balance sheet from {source_path_or_content or 'data/novacloud_fy2025_10k.md'}")

    def parse_architecture_diagram(self, source_path_or_content: str = "") -> ArchitectureTopology:
        """
        Parse architectural diagram into structured ArchitectureTopology schema.
        Extracts components, directional edges, communication protocols, and boundary perimeters
        from Mermaid or text files (data/novacloud_payment_arch.mmd by default).
        """
        content = ""
        if source_path_or_content:
            if os.path.exists(source_path_or_content):
                with open(source_path_or_content, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = source_path_or_content

        if not content:
            default_file = os.path.join(DATA_DIR, "novacloud_payment_arch.mmd")
            if os.path.exists(default_file):
                with open(default_file, "r", encoding="utf-8") as f:
                    content = f.read()

        if content:
            parsed = ArchitectureDiagramExtractor.extract_from_text(
                content,
                system_name="NovaCloud Global Payment Platform"
            )
            if parsed.nodes:
                return parsed

        raise FileNotFoundError(f"Could not load architecture diagram from {source_path_or_content or 'data/novacloud_payment_arch.mmd'}")

    # -----------------------------------------------------------------------
    # Live Multimodal Vision Extraction via Gemini API
    # -----------------------------------------------------------------------

    def extract_from_image_gemini(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        doc_type: str = "auto",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send image bytes directly to Gemini 2.0 Flash with structured JSON output schema.
        doc_type can be 'balance_sheet', 'architecture', or 'auto'.
        """
        key = api_key or self.api_key
        if not key:
            raise ValueError("A Gemini API Key is required for live multimodal vision inference.")

        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"

        prompt = (
            "You are a specialized multimodal document intelligence engine. "
            "Analyze this visual document. Determine whether it is a financial balance sheet / financial table, "
            "or an architectural system diagram.\n"
            "If it is a financial statement, return a JSON object with keys: "
            "'type': 'balance_sheet', 'company_name', 'reporting_period', 'currency', 'scale', and 'line_items' "
            "(each with 'name', 'value', 'currency', 'scale', 'category').\n"
            "If it is an architectural diagram, return a JSON object with keys: "
            "'type': 'architecture', 'system_name', 'environment', 'nodes' "
            "(each with 'id', 'name', 'component_type', 'zone', 'properties'), and 'edges' "
            "(each with 'source_id', 'target_id', 'protocol', 'sync_mode', 'description').\n"
            "Return valid JSON only."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": b64_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json"
            }
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text_out)

    def extract_from_pdf(self, pdf_path_or_bytes, doc_type: str = "balance_sheet") -> FinancialStatement:
        """Extract text from PDF pages using pypdf and parse into FinancialStatement."""
        if isinstance(pdf_path_or_bytes, bytes):
            stream = BytesIO(pdf_path_or_bytes)
        else:
            stream = open(pdf_path_or_bytes, "rb")

        reader = pypdf.PdfReader(stream)
        text_corpus = ""
        for page in reader.pages:
            text_corpus += (page.extract_text() or "") + "\n"

        if text_corpus.strip():
            parsed = FinancialTableExtractor.extract_from_text(text_corpus, company_name="Extracted PDF Entity")
            if parsed.line_items:
                return parsed

        raise ValueError(
            "Could not extract structured balance sheet line items from the provided PDF. "
            "Please ensure the document contains a readable tabular financial statement."
        )

    @staticmethod
    def dict_to_financial_statement(data: Dict[str, Any]) -> FinancialStatement:
        """Convert a structured dictionary (e.g. from Gemini Vision) into a typed FinancialStatement."""
        line_items = []
        for it in data.get("line_items", []):
            line_items.append(LineItem(
                name=it.get("name", "Unknown Item"),
                value=float(it.get("value", 0.0)),
                currency=it.get("currency", data.get("currency", "USD")),
                scale=it.get("scale", data.get("scale", "millions")),
                category=it.get("category", "General Assets")
            ))
        stmt = FinancialStatement(
            statement_type="balance_sheet",
            company_name=data.get("company_name", "Vision-Extracted Corporation"),
            reporting_period=data.get("reporting_period", "FY2025"),
            currency=data.get("currency", "USD"),
            scale=data.get("scale", "millions"),
            line_items=line_items
        )
        FinancialTableExtractor.compute_ratios(stmt)
        return stmt

    @staticmethod
    def dict_to_architecture_topology(data: Dict[str, Any]) -> ArchitectureTopology:
        """Convert a structured dictionary (e.g. from Gemini Vision) into a typed ArchitectureTopology."""
        nodes = []
        for n in data.get("nodes", []):
            nodes.append(DiagramNode(
                id=n.get("id", "comp").lower().replace(" ", "_"),
                name=n.get("name", "Component"),
                component_type=n.get("component_type", "microservice"),
                zone=n.get("zone", "Default Zone"),
                properties=n.get("properties", {})
            ))
        edges = []
        for e in data.get("edges", []):
            edges.append(DiagramEdge(
                source_id=e.get("source_id", "").lower().replace(" ", "_"),
                target_id=e.get("target_id", "").lower().replace(" ", "_"),
                protocol=e.get("protocol", "HTTPS"),
                sync_mode=e.get("sync_mode", "sync"),
                description=e.get("description", "")
            ))
        return ArchitectureTopology(
            system_name=data.get("system_name", "Vision-Extracted Architecture"),
            environment=data.get("environment", "Production"),
            nodes=nodes,
            edges=edges,
            boundary_zones=list(set(n.zone for n in nodes if n.zone)),
            description=f"Extracted topology with {len(nodes)} components and {len(edges)} paths."
        )

    # -----------------------------------------------------------------------
    # Showcase Datasets Loaded Authentically from Disk
    # -----------------------------------------------------------------------

    @staticmethod
    def load_apple_inc_showcase() -> Dict[str, Any]:
        """Loads and parses authentic Apple Inc. Form 10-K and PCC Architecture documents from data/."""
        apple_10k_file = os.path.join(DATA_DIR, "apple_fy2024_10k.md")
        apple_arch_file = os.path.join(DATA_DIR, "apple_pcc_arch.mmd")

        with open(apple_10k_file, "r", encoding="utf-8") as f:
            stmt = FinancialTableExtractor.extract_from_text(f.read(), company_name="Apple Inc", period="FY2024")
        with open(apple_arch_file, "r", encoding="utf-8") as f:
            topo = ArchitectureDiagramExtractor.extract_from_text(f.read(), system_name="Apple Private Cloud Compute & Ecosystem Infrastructure")

        return {"statement": stmt, "topology": topo}

    @staticmethod
    def _load_default_novacloud_balance_sheet() -> FinancialStatement:
        """Backwards compatibility wrapper to parse default 10-K document from disk."""
        parser = MultimodalVisionParser()
        return parser.parse_balance_sheet()

    @staticmethod
    def _load_default_payment_architecture() -> ArchitectureTopology:
        """Backwards compatibility wrapper to parse default architecture document from disk."""
        parser = MultimodalVisionParser()
        return parser.parse_architecture_diagram()
