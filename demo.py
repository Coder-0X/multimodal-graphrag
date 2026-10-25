"""
Interactive CLI Demonstration for Multimodal GraphRAG.
Demonstrates:
1. Multimodal document ingestion (10-K Balance Sheet + Architecture Diagram).
2. Knowledge Graph construction, PageRank centrality, and community detection.
3. LangGraph multi-hop reasoning solving complex cross-modal queries that traditional RAG fails on.
"""
import sys
import os
import io

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from src.pipeline import MultimodalGraphRAGPipeline

console = Console(force_terminal=True, legacy_windows=False)


def print_header():
    console.print(Panel.fit(
        "[bold cyan]MULTIMODAL GraphRAG ENGINE[/bold cyan]\n"
        "[dim]Cutting-Edge Applied AI & Complex Data Ingestion System[/dim]\n"
        "[yellow]Stack: NetworkX / Neo4j | LangGraph | Multimodal Vision Schemas[/yellow]",
        border_style="cyan"
    ))


def display_ingestion_summary(stats: dict):
    table = Table(title="[bold green]Knowledge Graph Ingestion & Analytics[/bold green]", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right", style="bold green")

    table.add_row("Total Entities (Nodes)", str(stats["total_entities"]))
    table.add_row("Total Relationships (Edges)", str(stats["total_relations"]))
    table.add_row("Cross-Modal Bridges (Finance <-> Infra)", str(stats["cross_modal_relations"]))
    table.add_row("Financial Metrics Extracted", str(stats["financial_metrics_count"]))
    table.add_row("Architecture Components Extracted", str(stats["architecture_components_count"]))
    table.add_row("Modular Communities Detected", str(stats["communities_detected"]))
    
    ratios = stats["computed_financial_ratios"]
    table.add_row("Computed Current Ratio", f"{ratios.get('current_ratio', 'N/A')}")
    table.add_row("Computed Quick Ratio", f"{ratios.get('quick_ratio', 'N/A')}")
    table.add_row("Computed Debt-to-Equity", f"{ratios.get('debt_to_equity_ratio', 'N/A')}")
    console.print(table)

    # Top Central Nodes (PageRank)
    pr_table = Table(title="[bold blue]Top Central Knowledge Graph Entities (PageRank)[/bold blue]", show_header=True)
    pr_table.add_column("Entity Name", style="bold")
    pr_table.add_column("PageRank Centrality", justify="right", style="cyan")
    for name, score in stats["top_central_entities"]:
        pr_table.add_row(name, f"{score:.4f}")
    console.print(pr_table)


def run_benchmark_queries(pipeline: MultimodalGraphRAGPipeline):
    benchmark_queries = [
        (
            "Query 1: Cross-Modal Financial Liability & Architecture Propagation",
            "If our AWS cloud hosting commitment liabilities spike due to payment gateway traffic surge, which system components are directly involved and what is the failover risk?"
        ),
        (
            "Query 2: Balance Sheet Liquidity Ratios & Cost Allocation",
            "What is the current ratio of the company, and which infrastructure cost centers contribute to the short-term operating obligations?"
        ),
        (
            "Query 3: Critical Path & Single Point of Failure (SPOF) Tracing",
            "Trace the multi-hop dependency path from the User Checkout flow down to the storage tier, identifying any single points of failure."
        )
    ]

    for title, q in benchmark_queries:
        console.print("\n" + "=" * 80, style="dim")
        console.print(f"[bold yellow]▶ {title}[/bold yellow]")
        console.print(f"[bold italic white]User Question: \"{q}\"[/bold italic white]\n")

        with console.status("[bold green]Executing LangGraph multi-hop reasoning workflow...[/bold green]"):
            result = pipeline.query(user_question=q, max_hops=3)

        # Show agent execution trace
        console.print("[dim]LangGraph Agent State Trace:[/dim]")
        for step in result.get("intermediate_reasoning", []):
            console.print(f"  [cyan]•[/cyan] [dim]{step}[/dim]")

        console.print("")
        # Display the synthesized answer
        console.print(Panel(
            Markdown(result["final_answer"]),
            title=f"[bold green]Resolved Response (Intent: {result['intent']})[/bold green]",
            border_style="green"
        ))


def main():
    print_header()

    console.print("[bold yellow]1. Initializing Ingestion Pipeline...[/bold yellow]")
    pipeline = MultimodalGraphRAGPipeline(use_neo4j=False)

    console.print("[bold yellow]2. Parsing Complex Documents (10-K Balance Sheet + Payment System Topology)...[/bold yellow]")
    stats = pipeline.ingest_documents()
    display_ingestion_summary(stats)

    console.print("\n[bold yellow]3. Executing Multi-Hop GraphRAG Benchmark Queries...[/bold yellow]")
    run_benchmark_queries(pipeline)

    console.print("\n[bold green]✓ Multimodal GraphRAG benchmark execution completed successfully![/bold green]\n")


if __name__ == "__main__":
    main()
