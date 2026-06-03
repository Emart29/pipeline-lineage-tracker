import asyncio
import sys
import webbrowser
from pathlib import Path

# Ensure project root is on sys.path when running as installed entry point
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table
from rich import box
from rich.tree import Tree

from db.base import AsyncSessionLocal
from store.metadata import MetadataStore
from store.blob import BlobStore
from core.dataset import DatasetTracker
from core.lineage import LineageGraph

console = Console()


def async_cmd(f):
    def wrapper(*args, **kwargs):
        asyncio.run(f(*args, **kwargs))
    wrapper.__name__ = f.__name__
    return wrapper


@click.group()
def cli():
    """Pipeline Lineage Tracker — dataset versioning and lineage tracing."""
    pass


@cli.command("datasets")
@async_cmd
async def cmd_datasets():
    """List all tracked dataset names."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        names = await meta.list_datasets()
    if not names:
        console.print("[yellow]No datasets tracked yet.[/yellow]")
        return
    table = Table(title="Tracked Datasets", box=box.ROUNDED, show_lines=True)
    table.add_column("Name", style="cyan bold")
    table.add_column("Versions", justify="right")
    table.add_column("Latest Hash", style="dim")
    table.add_column("Latest Created")
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        for name in names:
            versions = await meta.get_all_versions(name)
            latest = versions[-1] if versions else None
            table.add_row(name, str(len(versions)),
                latest.content_hash[:8] if latest else "-",
                latest.created_at.strftime("%Y-%m-%d %H:%M") if latest else "-")
    console.print(table)


@cli.command("versions")
@click.argument("dataset_name")
@async_cmd
async def cmd_versions(dataset_name):
    """List all versions of a dataset."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        versions = await meta.get_all_versions(dataset_name)
    if not versions:
        console.print(f"[yellow]No versions found for '{dataset_name}'.[/yellow]")
        return
    table = Table(title=f"Versions - {dataset_name}", box=box.ROUNDED, show_lines=True)
    table.add_column("Version", justify="right", style="cyan bold")
    table.add_column("Hash", style="dim")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    table.add_column("Created At")
    table.add_column("Description", style="dim")
    for v in versions:
        table.add_row(str(v.version), v.content_hash[:8], str(v.row_count),
            str(v.column_count), v.created_at.strftime("%Y-%m-%d %H:%M"), v.description or "")
    console.print(table)


@cli.command("diff")
@click.argument("dataset_name")
@click.argument("version_a", type=int)
@click.argument("version_b", type=int)
@async_cmd
async def cmd_diff(dataset_name, version_a, version_b):
    """Compare two dataset versions."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        blob = BlobStore()
        tracker = DatasetTracker(meta, blob)
        try:
            diff = await tracker.diff(dataset_name, version_a, version_b)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    console.print(f"\n[bold]Diff: {dataset_name} v{version_a} -> v{version_b}[/bold]\n")
    va, vb = diff["version_a_stats"], diff["version_b_stats"]
    d = diff["row_count_delta"]
    console.print(f"  Rows: {va['row_count']} -> {vb['row_count']} ({'green' if d>=0 else 'red'}[{d:+d}])")
    console.print(f"  Content identical: {diff['content_identical']}")
    sc = diff["schema_changes"]
    if sc["added"]:
        for col, dtype in sc["added"].items(): console.print(f"  [green]+ {col} ({dtype})[/green]")
    if sc["removed"]:
        for col, dtype in sc["removed"].items(): console.print(f"  [red]- {col} ({dtype})[/red]")
    if sc["changed"]:
        for col, ch in sc["changed"].items(): console.print(f"  [yellow]~ {col}: {ch['from']} -> {ch['to']}[/yellow]")
    if not any([sc["added"], sc["removed"], sc["changed"]]):
        console.print("  [dim]No schema changes[/dim]")
    console.print()


@cli.command("runs")
@click.option("--name", default=None)
@click.option("--limit", default=20, show_default=True)
@async_cmd
async def cmd_runs(name, limit):
    """List pipeline runs."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        runs = await meta.list_runs(name=name, limit=limit)
    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return
    colors = {"success": "green", "failed": "red", "running": "yellow"}
    table = Table(title="Pipeline Runs", box=box.ROUNDED, show_lines=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Name", style="cyan bold")
    table.add_column("Status")
    table.add_column("Git Commit", style="dim")
    table.add_column("Duration")
    table.add_column("Started At")
    for run in runs:
        c = colors.get(run.status, "white")
        dur = f"{run.duration_seconds:.1f}s" if run.duration_seconds else "-"
        table.add_row(str(run.run_number), run.name, f"[{c}]{run.status}[/{c}]",
            run.git_commit[:8] if run.git_commit else "-", dur,
            run.started_at.strftime("%Y-%m-%d %H:%M"))
    console.print(table)


@cli.command("trace")
@click.argument("node_id")
@click.option("--type", "node_type", default="dataset",
    type=click.Choice(["dataset", "pipeline_run", "model", "prediction"]))
@async_cmd
async def cmd_trace(node_id, node_type):
    """Trace full upstream lineage of a node."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        lg = LineageGraph(meta)
        result = await lg.trace_upstream(node_id, node_type)
    path = result["path"]
    if not path:
        console.print(f"[yellow]No lineage found for {node_type}:{node_id}[/yellow]")
        return
    console.print(f"\n[bold]Upstream lineage - depth {result['depth']}[/bold]\n")
    tree = Tree(f"[bold cyan]{path[0]}[/bold cyan]")
    node = tree
    for step in path[1:]:
        node = node.add(f"[cyan]{step}[/cyan]")
    console.print(tree)
    console.print()


@cli.command("impact")
@click.argument("dataset_name")
@click.option("--version", default=None, type=int)
@async_cmd
async def cmd_impact(dataset_name, version):
    """Impact analysis - what breaks if this dataset changes?"""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        lg = LineageGraph(meta)
        result = await lg.impact_analysis(dataset_name, version)
    ver_label = f"v{version}" if version else "latest"
    console.print(f"\n[bold]Impact Analysis - {dataset_name} ({ver_label})[/bold]\n")
    risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
    risk = result["risk_level"]
    console.print(f"  Risk level: [{risk_colors[risk]}]{risk.upper()}[/{risk_colors[risk]}]")
    console.print(f"  Affected pipeline runs: {len(result['affected_pipeline_runs'])}")
    console.print(f"  Affected models:        {len(result['affected_models'])}")
    console.print(f"  Affected predictions:   {result['affected_predictions_count']}")
    if result["affected_models"]:
        console.print("\n  [bold]Models at risk:[/bold]")
        for m in result["affected_models"]:
            console.print(f"    * {m.name} v{m.version} ({m.framework})")
    console.print()


@cli.command("graph")
@click.option("--output", default="lineage.html", show_default=True)
@click.option("--open", "open_browser", is_flag=True)
@async_cmd
async def cmd_graph(output, open_browser):
    """Export the full lineage DAG as an interactive HTML file."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        lg = LineageGraph(meta)
        G = await lg.build_graph()
        html = await lg.to_pyvis_html(G)
    Path(output).write_text(html, encoding="utf-8")
    console.print(f"[green]Graph exported to {output}[/green] ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    if open_browser:
        webbrowser.open(f"file://{Path(output).resolve()}")


@cli.command("stats")
@async_cmd
async def cmd_stats():
    """Summary statistics of the lineage graph."""
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        lg = LineageGraph(meta)
        stats = await lg.get_stats()
    table = Table(title="Lineage Graph Stats", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")
    for label, value in [
        ("Total nodes", str(stats["total_nodes"])),
        ("Total edges", str(stats["total_edges"])),
        ("Dataset versions", str(stats["dataset_versions"])),
        ("Pipeline runs", str(stats["pipeline_runs"])),
        ("Models", str(stats["models"])),
        ("Predictions", str(stats["predictions"])),
        ("Avg lineage depth", str(stats["avg_lineage_depth"])),
    ]:
        table.add_row(label, value)
    console.print(table)


if __name__ == "__main__":
    cli()
