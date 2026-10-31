"""
Grounded LLM Reasoning Client for LangGraph GraphRAG.
Supports:
1. Live Gemini API (gemini-2.0-flash / gemini-1.5-flash)
2. Live OpenAI API (gpt-4o / gpt-4o-mini)
3. Deterministic Extractive Graph-Reasoner (Zero-Key Local Fallback)
Eliminates static canned text by dynamically synthesizing answers directly
from retrieved Knowledge Graph subgraphs and multi-hop paths.
"""
from typing import Dict, Any, List, Optional
import os
import json
import httpx


class LLMReasoningClient:
    """
    Manages genuine model inference over Knowledge Graph evidence subgraphs.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        self.gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.provider = provider

    @property
    def is_live_llm(self) -> bool:
        return bool(self.gemini_key or self.openai_key)

    def generate_synthesis(
        self,
        query: str,
        intent: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        paths: List[List[str]],
        evidence_text: str
    ) -> Dict[str, Any]:
        """Generate grounded multi-hop response using live LLM or extractive graph engine."""
        if self.gemini_key and (self.provider in ("auto", "gemini")):
            return self._call_gemini(query, intent, entities, relations, paths, evidence_text)
        elif self.openai_key and (self.provider in ("auto", "openai")):
            return self._call_openai(query, intent, entities, relations, paths, evidence_text)
        else:
            return self._extractive_graph_synthesis(query, intent, entities, relations, paths)

    def _call_gemini(
        self,
        query: str,
        intent: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        paths: List[List[str]],
        evidence_text: str
    ) -> Dict[str, Any]:
        """Execute inference against Google Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
        
        prompt = f"""You are a GraphRAG Multi-Hop Reasoning Engine.
User Query: {query}
Classified Intent: {intent}

Retrieved Knowledge Graph Evidence:
{evidence_text}

Multi-Hop Paths:
{json.dumps(paths, indent=2)}

Instructions:
1. Answer the user query thoroughly and accurately using ONLY the provided Knowledge Graph evidence.
2. Directly answer specific questions (e.g. ratios, component architecture, single points of failure, failovers).
3. Explicitly cite the graph triples and paths used in your deduction.
4. Format your answer with clean Markdown headers, bullet points, and citations.
"""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024}
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                answer = data["candidates"][0]["content"]["parts"][0]["text"]
                return {
                    "answer": answer,
                    "engine": "Gemini 2.0 Flash (Live LLM)",
                    "citations": [f"({r['source_id']}) -[{r['relation_type']}]-> ({r['target_id']})" for r in relations[:8]]
                }
        except Exception as e:
            # Fallback to extractive if network or API key fails
            res = self._extractive_graph_synthesis(query, intent, entities, relations, paths)
            res["engine"] = f"Extractive Engine (Gemini API error: {e})"
            return res

    def _call_openai(
        self,
        query: str,
        intent: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        paths: List[List[str]],
        evidence_text: str
    ) -> Dict[str, Any]:
        """Execute inference against OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        prompt = f"""You are a GraphRAG Multi-Hop Reasoning Engine.
User Query: {query}
Intent: {intent}

Evidence Subgraph:
{evidence_text}

Answer the query grounded strictly on the graph evidence with explicit citations."""
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                return {
                    "answer": answer,
                    "engine": "GPT-4o-mini (Live LLM)",
                    "citations": [f"({r['source_id']}) -[{r['relation_type']}]-> ({r['target_id']})" for r in relations[:8]]
                }
        except Exception as e:
            res = self._extractive_graph_synthesis(query, intent, entities, relations, paths)
            res["engine"] = f"Extractive Engine (OpenAI API error: {e})"
            return res

    def _extractive_graph_synthesis(
        self,
        query: str,
        intent: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        paths: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Deterministic, dynamic extractive graph reasoning when no API key is provided.
        Constructs factual answers derived strictly from retrieved entity properties and paths,
        with ZERO canned hardcoded text.
        """
        ent_map = {e["id"]: e for e in entities}
        lines: List[str] = []

        lines.append(f"### Graph-Grounded Analysis: \"{query}\"\n")
        lines.append(f"**Execution Mode**: `Extractive Graph-Reasoner` (Deterministic Local Engine)")
        lines.append(f"**Classified Intent**: `{intent}` | **Evidence Subgraph**: `{len(entities)}` entities, `{len(relations)}` relations\n")

        # 1. Multi-hop traversal paths
        if paths:
            lines.append("#### Multi-Hop Associative Pathways:")
            for p in paths:
                named_path = [ent_map.get(nid, {}).get("name", nid) for nid in p]
                lines.append(f"```text\n{' ──▶ '.join(named_path)}\n```")
            lines.append("")

        # 2. Extract domain findings
        financial_items = [e for e in entities if e.get("entity_type") == "FINANCIAL_METRIC"]
        system_items = [e for e in entities if e.get("entity_type") == "SYSTEM_COMPONENT"]
        zones = [e for e in entities if e.get("entity_type") == "INFRASTRUCTURE_ZONE"]

        if financial_items:
            lines.append("#### Financial Metrics & Balance Sheet Evidence:")
            for item in financial_items:
                props = item.get("properties", {})
                val = props.get("value")
                scale = props.get("scale", "")
                curr = props.get("currency", "")
                if val is not None:
                    lines.append(f"- **{item['name']}**: `{val:,.2f}` {curr} ({scale}) — *{item.get('description', '')}*")
                else:
                    lines.append(f"- **{item['name']}**: *{item.get('description', '')}*")
            lines.append("")

        if system_items:
            lines.append("#### Infrastructure Components & Flow Evidence:")
            for item in system_items:
                props = item.get("properties", {})
                ctype = props.get("component_type", "microservice")
                zone = props.get("zone", "VPC")
                lines.append(f"- **{item['name']}** (`{ctype}` in `{zone}`): {item.get('description', '')}")
            lines.append("")

        # 3. Dynamic Topology & Risk Deductions
        lines.append("#### Graph Topology Deductions & Evidence Synthesis:")
        
        # Check for cross-modal links
        cross_links = [r for r in relations if "FUNDS" in r.get("relation_type", "") or "HOSTS" in r.get("relation_type", "") or "ASSOCIATED" in r.get("relation_type", "")]
        if cross_links:
            for cl in cross_links:
                src_name = ent_map.get(cl["source_id"], {}).get("name", cl["source_id"])
                tgt_name = ent_map.get(cl["target_id"], {}).get("name", cl["target_id"])
                lines.append(f"- **Cross-Modal Link**: `{src_name}` ──[{cl['relation_type']}]──▶ `{tgt_name}` ({cl.get('description', '')})")

        # Check for critical paths
        crit_links = [r for r in relations if r.get("weight", 1.0) > 1.2 or "critical" in r.get("description", "").lower()]
        if crit_links:
            for cr in crit_links:
                src_name = ent_map.get(cr["source_id"], {}).get("name", cr["source_id"])
                tgt_name = ent_map.get(cr["target_id"], {}).get("name", cr["target_id"])
                lines.append(f"- **Critical Dependency**: `{src_name}` communicates with `{tgt_name}` [{cr.get('relation_type', '')}].")

        # Fallback if no specific links
        if not cross_links and not crit_links and relations:
            for r in relations[:4]:
                src_name = ent_map.get(r["source_id"], {}).get("name", r["source_id"])
                tgt_name = ent_map.get(r["target_id"], {}).get("name", r["target_id"])
                lines.append(f"- `{src_name}` connects to `{tgt_name}` via `{r.get('relation_type', '')}`")

        lines.append("\n#### Subgraph Citations:")
        citations = []
        for r in relations[:8]:
            src = ent_map.get(r["source_id"], {}).get("name", r["source_id"])
            tgt = ent_map.get(r["target_id"], {}).get("name", r["target_id"])
            c = f"({src}) ──[{r['relation_type']}]──▶ ({tgt})"
            lines.append(f"- {c}")
            citations.append(c)

        return {
            "answer": "\n".join(lines),
            "engine": "Extractive Graph-Reasoner (Deterministic Local Engine)",
            "citations": citations
        }
