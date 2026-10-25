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
                    "footnotes": item.footnotes
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
                properties={"ratio": ratio_name, "value": ratio_val, "type": "derived_metric"}
            ))
            relations.append(KGRelation(
                source_id=company_id,
                target_id=ratio_id,
                relation_type="HAS_FINANCIAL_RATIO",
                description=f"{statement.company_name} has {display_name} of {ratio_val}"
            ))

        return entities, relations
