"""
Financial table extractor & ratio analyzer.
Transforms tabular financial statements (e.g. balance sheets) into typed models
and produces Knowledge Graph entities and relations.
"""
from typing import List, Dict, Tuple, Optional
import re
from .schemas import FinancialStatement, LineItem, KGEntity, KGRelation


class FinancialTableExtractor:
    """Extracts, normalizes, and computes financial metrics from balance sheets and tables."""

    @classmethod
    def extract_from_text(
        cls,
        raw_text: str,
        company_name: str = "Extracted Entity",
        period: str = "FY2025"
    ) -> FinancialStatement:
        """
        Parses raw text, markdown tables, or PDF extractions into a normalized FinancialStatement.
        Dynamically extracts line items, monetary values, scales, and categories.
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        line_items: List[LineItem] = []
        current_category = "General Assets"
        scale = "thousands" if "thousand" in raw_text.lower() else ("millions" if "million" in raw_text.lower() else "units")
        currency = "USD"
        if "$" in raw_text or "USD" in raw_text:
            currency = "USD"
        elif "€" in raw_text or "EUR" in raw_text:
            currency = "EUR"
        elif "£" in raw_text or "GBP" in raw_text:
            currency = "GBP"

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            line_lower = line_clean.lower()

            # Skip table delimiter rows (e.g. |:---|---:| or --------)
            if re.match(r'^[:|\-\s]+$', line_clean):
                continue

            # Detect category headings
            if any(h in line_lower for h in ["current assets", "assets", "liabilities and stockholders", "current liabilities", "non-current", "equity"]):
                if "current asset" in line_lower and "total" not in line_lower:
                    current_category = "Current Assets"
                elif ("non-current asset" in line_lower or "property" in line_lower) and "total" not in line_lower:
                    current_category = "Non-Current Assets"
                elif "current liabilit" in line_lower and "total" not in line_lower:
                    current_category = "Current Liabilities"
                elif ("non-current liabilit" in line_lower or "long-term debt" in line_lower) and "total" not in line_lower:
                    current_category = "Non-Current Liabilities"
                elif ("equity" in line_lower or "stockholder" in line_lower or "shareholder" in line_lower) and "total" not in line_lower:
                    current_category = "Stockholders Equity"
                elif "total asset" in line_lower and not any(char.isdigit() for char in line_clean):
                    current_category = "Total Assets"
                elif "total liabilit" in line_lower and not any(char.isdigit() for char in line_clean):
                    current_category = "Total Liabilities"

            # Parse line item name and numeric value
            raw_name: Optional[str] = None
            val: Optional[float] = None

            if '|' in line_clean:
                cells = [c.strip() for c in line_clean.split('|') if c.strip()]
                if len(cells) >= 2:
                    potential_name = cells[0]
                    # Check that first cell isn't a table header title
                    if not any(hdr in potential_name.lower() for hdr in ["balance sheet", "statement of", "as of", "consolidated", "item", "description"]):
                        for candidate_cell in reversed(cells[1:]):
                            num_m = re.search(r'\(?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\)?', candidate_cell)
                            if num_m:
                                # Ensure candidate_cell is not a year header like "2025" without currency if preceded by month
                                if re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', candidate_cell, re.I):
                                    continue
                                num_str = num_m.group(1).replace(',', '')
                                try:
                                    val = float(num_str)
                                    if '(' in candidate_cell and ')' in candidate_cell:
                                        val = -val
                                    raw_name = potential_name
                                    break
                                except ValueError:
                                    pass
            else:
                # Colon or dot/space delimited: e.g. "Cash and equivalents: $5,000,000" or "Total Assets ... 10,000"
                parts = re.split(r'[:\t]|\.{2,}', line_clean)
                if len(parts) >= 2:
                    potential_name = parts[0].strip()
                    if not any(hdr in potential_name.lower() for hdr in ["balance sheet", "statement of", "as of"]):
                        num_m = re.search(r'\(?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\)?', parts[-1])
                        if num_m:
                            num_str = num_m.group(1).replace(',', '')
                            try:
                                val = float(num_str)
                                if '(' in parts[-1] and ')' in parts[-1]:
                                    val = -val
                                raw_name = potential_name
                            except ValueError:
                                pass
                else:
                    # Fallback regex for unstructured lines: "Cash and cash equivalents $5,000,000"
                    m = re.search(r'^([A-Za-z0-9\s,\'&\-\(\)\/]+?)\s+[\$€£]?\(?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\)?$', line_clean)
                    if m and not any(hdr in m.group(1).lower() for hdr in ["balance sheet", "statement of", "as of"]):
                        try:
                            val = float(m.group(2).replace(',', ''))
                            if '(' in line_clean and ')' in line_clean:
                                val = -val
                            raw_name = m.group(1).strip()
                        except ValueError:
                            pass

            if raw_name and val is not None:
                # Refine category for explicit totals
                cat = current_category
                raw_name_lower = raw_name.lower()
                if "total current asset" in raw_name_lower:
                    cat = "Current Assets"
                elif "total current liabilit" in raw_name_lower:
                    cat = "Current Liabilities"
                elif "total asset" in raw_name_lower:
                    cat = "Total Assets"
                elif "total liabilit" in raw_name_lower:
                    cat = "Total Liabilities"
                elif "total stockholder" in raw_name_lower or "total equity" in raw_name_lower:
                    cat = "Stockholders Equity"

                line_items.append(LineItem(
                    name=raw_name,
                    value=val,
                    currency=currency,
                    scale=scale,
                    category=cat
                ))

        # Correlate and attach footnotes / notes from document body (e.g. Note 4, Note 8)
        note_sections = re.findall(r'(?:Note\s+\d+|###+\s*Note\s+\d+)[—\-\:\s]+([^\n]+(?:\n[^\n#]+)*)', raw_text, re.IGNORECASE)
        for note_text in note_sections:
            note_lower = note_text.lower()
            for item in line_items:
                keywords = [k for k in item.name.lower().split() if len(k) > 4 and k not in ("total", "other", "under")]
                if any(kw in note_lower for kw in keywords):
                    item.footnotes.append(note_text.strip())

        statement = FinancialStatement(
            statement_type="balance_sheet",
            company_name=company_name,
            reporting_period=period,
            currency=currency,
            scale=scale,
            line_items=line_items,
            raw_markdown=raw_text
        )
        cls.compute_ratios(statement)
        return statement

    @staticmethod
    def compute_ratios(statement: FinancialStatement) -> Dict[str, float]:
        """Compute key liquidity and leverage ratios from balance sheet line items."""
        # Normalize keys by removing punctuation
        def clean_key(s: str) -> str:
            return re.sub(r'[^a-z0-9]', '', s.lower())

        items_by_name = {clean_key(item.name): item.value for item in statement.line_items}
        
        # Helper to find values with fuzzy normalized matching
        def find_val(*patterns: str) -> Optional[float]:
            for pat in patterns:
                cpat = clean_key(pat)
                for k, v in items_by_name.items():
                    if cpat in k or k in cpat:
                        return v
            return None

        ratios: Dict[str, float] = {}

        current_assets = find_val("total current assets")
        current_liabilities = find_val("total current liabilities")
        total_assets = find_val("total assets")
        total_liabilities = find_val("total liabilities")
        stockholders_equity = find_val("total stockholders equity", "total shareholders equity", "total equity")
        cash_and_equiv = find_val("cash and cash equivalents", "cash & cash equivalents")
        marketable_securities = find_val("marketable securities", "short-term investments") or 0.0
        accounts_receivable = find_val("accounts receivable") or 0.0

        if current_assets is not None and current_liabilities and current_liabilities > 0:
            ratios["current_ratio"] = round(current_assets / current_liabilities, 3)
            ratios["working_capital"] = round(current_assets - current_liabilities, 2)
            
            # Quick Ratio = (Cash + Securities + AR) / Current Liabilities
            if cash_and_equiv is not None:
                quick_assets = cash_and_equiv + marketable_securities + accounts_receivable
                ratios["quick_ratio"] = round(quick_assets / current_liabilities, 3)

        if total_liabilities is not None and stockholders_equity and stockholders_equity > 0:
            ratios["debt_to_equity_ratio"] = round(total_liabilities / stockholders_equity, 3)

        if total_assets is not None and total_liabilities is not None and total_assets > 0:
            ratios["debt_to_assets_ratio"] = round(total_liabilities / total_assets, 3)

        statement.computed_metrics = ratios
        return ratios

    @classmethod
    def to_knowledge_graph(cls, statement: FinancialStatement) -> Tuple[List[KGEntity], List[KGRelation]]:
        """Converts a parsed FinancialStatement into KG entities and relationships."""
        entities: List[KGEntity] = []
        relations: List[KGRelation] = []

        # 1. Company Entity
        company_id = f"org_{statement.company_name.lower().replace(' ', '_')}"
        entities.append(KGEntity(
            id=company_id,
            name=statement.company_name,
            entity_type="ORGANIZATION",
            description=f"Corporate entity reporting financial statements for {statement.reporting_period}",
            properties={"reporting_period": statement.reporting_period, "currency": statement.currency}
        ))

        # 2. Fiscal Period Entity
        period_id = f"period_{statement.reporting_period.lower()}"
        entities.append(KGEntity(
            id=period_id,
            name=statement.reporting_period,
            entity_type="FISCAL_PERIOD",
            description=f"Fiscal reporting period {statement.reporting_period}",
            properties={"period": statement.reporting_period}
        ))

        # Link Company -> Fiscal Period
        relations.append(KGRelation(
            source_id=company_id,
            target_id=period_id,
            relation_type="REPORTS_FOR",
            description=f"{statement.company_name} published statement for {statement.reporting_period}"
        ))

        # 3. Line Items and Category Aggregations
        category_ids: Dict[str, str] = {}
        for item in statement.line_items:
            cat_clean = item.category.strip()
            cat_id = f"cat_{cat_clean.lower().replace(' ', '_')}"
            
            if cat_clean not in category_ids:
                category_ids[cat_clean] = cat_id
                entities.append(KGEntity(
                    id=cat_id,
                    name=cat_clean,
                    entity_type="CONCEPT",
                    description=f"Accounting category: {cat_clean}",
                    properties={"category": cat_clean}
                ))
                # Link Company -> Category
                relations.append(KGRelation(
                    source_id=company_id,
                    target_id=cat_id,
                    relation_type="HAS_ACCOUNTING_SECTION",
                    description=f"Statement contains section {cat_clean}"
                ))

            # Metric Entity
            item_slug = re.sub(r'[^a-zA-Z0-9_]', '', item.name.lower().replace(' ', '_'))
            metric_id = f"metric_{item_slug}_{statement.reporting_period.lower()}"
            entities.append(KGEntity(
                id=metric_id,
                name=f"{item.name} ({statement.reporting_period})",
                entity_type="FINANCIAL_METRIC",
                description=f"{item.name} valued at {item.value:,} {statement.currency} ({item.scale}) for {statement.reporting_period}",
                properties={
                    "metric_name": item.name,
                    "value": item.value,
                    "currency": statement.currency,
                    "scale": item.scale,
                    "category": item.category,
                    "footnotes": item.footnotes,
                    "subtype": "line_item"
                }
            ))

            # Relation: Metric -> Category (PART_OF)
            relations.append(KGRelation(
                source_id=metric_id,
                target_id=cat_id,
                relation_type="PART_OF",
                description=f"'{item.name}' is categorized under '{cat_clean}'"
            ))

            # Relation: Metric -> Fiscal Period (RECORDED_IN)
            relations.append(KGRelation(
                source_id=metric_id,
                target_id=period_id,
                relation_type="RECORDED_IN",
                description=f"Recorded during fiscal period {statement.reporting_period}"
            ))

        # 4. Computed Ratios as Entities
        cls.compute_ratios(statement)
        for ratio_name, ratio_val in statement.computed_metrics.items():
            ratio_id = f"ratio_{ratio_name}_{statement.reporting_period.lower()}"
            display_name = ratio_name.replace('_', ' ').title()
            entities.append(KGEntity(
                id=ratio_id,
                name=f"{display_name} ({statement.reporting_period})",
                entity_type="FINANCIAL_METRIC",
                description=f"Computed ratio {display_name} = {ratio_val}",
                properties={"ratio": ratio_name, "value": ratio_val, "subtype": "derived_ratio", "type": "derived_metric"}
            ))
            relations.append(KGRelation(
                source_id=company_id,
                target_id=ratio_id,
                relation_type="HAS_FINANCIAL_RATIO",
                description=f"{statement.company_name} has {display_name} of {ratio_val}"
            ))

        return entities, relations
